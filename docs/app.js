// mlrecipe Explorer — pure GitHub-driven, no backend.
//
// Three panes: Browse (inspect a single recipe), Search (find published
// recipes via the GitHub Search API), Publish (copy-paste guide).
//
// Reads recipe metadata two ways:
//   - Browse: a tagged GitHub Release that has a .toml asset attached.
//             This is what `mlrecipe push` produces today.
//   - Search: the GitHub Code Search API for `recipe.toml` files,
//             paged anonymously. We resolve each hit to its raw blob
//             and parse just enough TOML to render a one-line summary.

const $ = (id) => document.getElementById(id);

// ---------- minimal TOML parser ----------
// Subset we emit: top-level keys, [section], [[array.section]], strings
// (basic), ints, floats, bools, arrays of strings.
function parseTOML(text) {
  const lines = text.split(/\r?\n/);

  // Fold multi-line arrays into a single logical line.
  const folded = [];
  let buf = null;
  let bracketDepth = 0;
  for (const raw of lines) {
    if (buf !== null) {
      buf += " " + raw;
      for (const c of raw) {
        if (c === "[") bracketDepth++;
        else if (c === "]") bracketDepth--;
      }
      if (bracketDepth <= 0) { folded.push(buf); buf = null; bracketDepth = 0; }
      continue;
    }
    const eq = raw.indexOf("=");
    const lbr = raw.indexOf("[");
    if (eq >= 0 && lbr > eq) {
      let depth = 0, opened = false;
      for (let i = lbr; i < raw.length; i++) {
        if (raw[i] === "[") { depth++; opened = true; }
        else if (raw[i] === "]") depth--;
      }
      if (opened && depth > 0) { buf = raw; bracketDepth = depth; continue; }
    }
    folded.push(raw);
  }
  if (buf !== null) folded.push(buf);

  const root = {};
  let cur = root;
  function tableAt(obj, path) {
    for (const p of path) { if (!(p in obj)) obj[p] = {}; obj = obj[p]; }
    return obj;
  }
  function parseValue(s) {
    s = s.trim();
    if (s.startsWith('"') && s.endsWith('"')) return JSON.parse(s);
    if (s === "true") return true;
    if (s === "false") return false;
    if (s.startsWith("[") && s.endsWith("]")) {
      const inner = s.slice(1, -1).trim();
      if (!inner) return [];
      const items = []; let depth = 0, q = false, start = 0;
      for (let i = 0; i <= inner.length; i++) {
        const c = inner[i];
        if (c === '"' && inner[i-1] !== "\\") q = !q;
        if (!q && c === "[") depth++;
        if (!q && c === "]") depth--;
        if (!q && depth === 0 && (c === "," || i === inner.length)) {
          const tok = inner.slice(start, i).trim();
          if (tok) items.push(parseValue(tok));
          start = i + 1;
        }
      }
      return items;
    }
    if (/^-?\d+$/.test(s)) return parseInt(s, 10);
    if (/^-?\d+\.\d+(e-?\d+)?$/i.test(s)) return parseFloat(s);
    return s;
  }
  for (const raw of folded) {
    const line = raw.split("#")[0].trim();
    if (!line) continue;
    if (line.startsWith("[[") && line.endsWith("]]")) {
      const path = line.slice(2, -2).split(".").map((p) => p.trim());
      const parent = tableAt(root, path.slice(0, -1));
      const last = path[path.length - 1];
      if (!Array.isArray(parent[last])) parent[last] = [];
      const entry = {};
      parent[last].push(entry);
      cur = entry;
      continue;
    }
    if (line.startsWith("[") && line.endsWith("]") && !line.includes("=")) {
      const path = line.slice(1, -1).split(".").map((p) => p.trim());
      cur = tableAt(root, path);
      continue;
    }
    const eq = line.indexOf("=");
    if (eq < 0) continue;
    cur[line.slice(0, eq).trim()] = parseValue(line.slice(eq + 1));
  }
  return root;
}

// ---------- formatting helpers ----------
function fmtBytes(n) {
  if (n == null) return "?";
  if (n < 1024) return n + " B";
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
  if (n < 1024 ** 3) return (n / 1024 / 1024).toFixed(1) + " MB";
  return (n / 1024 ** 3).toFixed(2) + " GB";
}

function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

// Approximate base-model size in bytes. Used to compute the recipe
// compression ratio. We don't go to the network for this — the table
// covers everything you'd realistically host as a recipe today, and
// unknown bases just render without a ratio.
const BASE_SIZE_HINTS = {
  "gpt2": 500e6,
  "gpt2-medium": 1.5e9,
  "gpt2-large": 3.1e9,
  "gpt2-xl": 6.4e9,
  "distilgpt2": 330e6,
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 2.2e9,
  "Qwen/Qwen2.5-0.5B": 1e9,
  "Qwen/Qwen2.5-0.5B-Instruct": 1e9,
  "Qwen/Qwen2.5-1.5B": 3e9,
  "Qwen/Qwen2.5-1.5B-Instruct": 3e9,
  "Qwen/Qwen2.5-3B": 6.2e9,
  "Qwen/Qwen2.5-7B": 15e9,
  "Qwen/Qwen2.5-7B-Instruct": 15e9,
  "meta-llama/Llama-3-8B": 16e9,
  "meta-llama/Llama-3-8B-Instruct": 16e9,
  "meta-llama/Meta-Llama-3-8B": 16e9,
  "meta-llama/Llama-3.1-8B": 16e9,
  "meta-llama/Llama-3.1-8B-Instruct": 16e9,
  "mistralai/Mistral-7B-v0.1": 14e9,
  "mistralai/Mistral-7B-Instruct-v0.2": 14e9,
  "google/gemma-2b": 5e9,
  "google/gemma-7b": 14e9,
  "microsoft/Phi-3-mini-4k-instruct": 7.6e9,
};
function estimateBaseSize(ref) {
  if (!ref) return null;
  if (BASE_SIZE_HINTS[ref]) return BASE_SIZE_HINTS[ref];
  for (const [k, v] of Object.entries(BASE_SIZE_HINTS)) {
    if (ref.startsWith(k)) return v;
  }
  return null;
}

// ---------- GitHub API ----------
async function fetchRelease(repo, tag) {
  const url = tag === "latest"
    ? `https://api.github.com/repos/${repo}/releases/latest`
    : `https://api.github.com/repos/${repo}/releases/tags/${tag}`;
  const r = await fetch(url, { headers: { Accept: "application/vnd.github+json" } });
  if (r.status === 404) throw new Error(`Release not found: ${repo}@${tag}`);
  if (r.status === 403) throw new Error("GitHub API rate-limited. Sign in to GitHub in another tab and try again.");
  if (!r.ok) throw new Error(`GitHub API error ${r.status}`);
  return r.json();
}

async function fetchTomlAssetText(release) {
  const tomlAsset = release.assets.find((a) => a.name.endsWith(".toml"));
  if (!tomlAsset) throw new Error("No .toml asset on this release. Was it pushed by an old mlrecipe version?");
  const r = await fetch(tomlAsset.browser_download_url);
  if (!r.ok) throw new Error(`Fetch toml: ${r.status}`);
  return r.text();
}

// ---------- Browse: render a single recipe ----------
function renderRecipe(repo, tag, release, recipe) {
  const bundle = release.assets.find((a) => a.name.endsWith(".tar.gz"));
  const bundleSize = bundle?.size ?? 0;

  const meta = recipe.recipe || {};
  const base = recipe.base || {};
  const adapters = recipe.adapters || [];
  const training = recipe.training || {};

  const baseSize = estimateBaseSize(base.ref);
  const ratio = baseSize && bundleSize ? Math.round(baseSize / bundleSize) : null;

  const html = [];
  html.push(`<div class="recipe-name">${escapeHTML(meta.name || "recipe")}</div>`);
  html.push(`<p class="recipe-base">Base: <code>${escapeHTML(base.ref || "?")}</code>${
    base.revision ? ` <span class="meta">@ ${escapeHTML(base.revision)}</span>` : ""
  }</p>`);

  // KV table: format, release, published, training method.
  html.push(`<div class="kv">`);
  html.push(`<div class="k">Format</div><div class="v"><code>${escapeHTML(meta.version || "?")}</code></div>`);
  html.push(`<div class="k">Release</div><div class="v"><a href="https://github.com/${repo}/releases/tag/${tag}" target="_blank">${escapeHTML(repo)}@${escapeHTML(tag)}</a></div>`);
  if (release.published_at || release.created_at) {
    html.push(`<div class="k">Published</div><div class="v">${new Date(release.published_at || release.created_at).toISOString().slice(0, 10)}</div>`);
  }
  if (training.method) html.push(`<div class="k">Method</div><div class="v">${escapeHTML(training.method)}</div>`);
  html.push(`</div>`);

  // Size figure: bundle vs estimated merged checkpoint.
  if (bundleSize) {
    html.push(`<div class="size-figure">`);
    html.push(`<div class="size-col"><h3>Recipe bundle</h3><div class="size-num">${fmtBytes(bundleSize)}</div></div>`);
    if (ratio) {
      html.push(`<div class="size-mid"><div class="x">${ratio}×</div><div class="x-label">smaller</div></div>`);
      html.push(`<div class="size-col right"><h3>Merged checkpoint (estimate)</h3><div class="size-num">~${fmtBytes(baseSize)}</div></div>`);
    } else {
      html.push(`<div class="size-mid"><div class="x">—</div><div class="x-label">unknown base</div></div>`);
      html.push(`<div class="size-col right"><h3>Merged checkpoint</h3><div class="size-num" style="color:var(--text-muted);font-size:1rem">unknown</div></div>`);
    }
    html.push(`</div>`);
  }

  // Adapters list.
  if (adapters.length) {
    html.push(`<h2>Adapter${adapters.length > 1 ? "s" : ""}</h2>`);
    adapters.forEach((a, i) => {
      html.push(`<div class="adapter-row">`);
      const hashShort = a.artifact ? a.artifact.slice(0, 24) + "…" : "?";
      html.push(`<div class="head"><span class="typ">${escapeHTML(a.type || "?")}</span><span class="hash">${escapeHTML(hashShort)}</span></div>`);
      const cfg = [];
      if (a.rank != null) cfg.push(`rank ${a.rank}`);
      if (a.alpha != null) cfg.push(`α ${a.alpha}`);
      if (a.lora_dropout) cfg.push(`dropout ${a.lora_dropout}`);
      if (a.bias && a.bias !== "none") cfg.push(`bias ${a.bias}`);
      if (a.fan_in_fan_out) cfg.push("Conv1D");
      if (cfg.length) html.push(`<div class="config">${cfg.join(" · ")}</div>`);
      if (a.target_modules?.length) {
        html.push(`<div class="targets">Targets: ${a.target_modules.map((m) => `<code>${escapeHTML(m)}</code>`).join(" ")}</div>`);
      }
      html.push(`</div>`);
    });
  }

  // Use it.
  html.push(`<h2>Use this recipe</h2>`);
  html.push(`<div class="cmds">
<span class="prompt">$</span>pip install git+https://github.com/shiahonb777/mlrecipe.git
<span class="prompt">$</span>mlrecipe clone ${escapeHTML(repo)}@${escapeHTML(tag)}
<span class="prompt">$</span>cd ${escapeHTML(repo.split("/").pop())}
<span class="prompt">$</span>mlrecipe materialize ./merged   <span class="comment"># bit-equal to peft.merge_and_unload()</span></div>`);
  html.push(`<p class="meta" style="margin-top:14px">Or <a class="accent" href="run.html?ref=${encodeURIComponent(repo + "@" + tag)}">run it in the browser</a> for small base models.</p>`);

  $("browse-result").innerHTML = html.join("");
}

async function inspectRef(refStr) {
  const target = $("browse-result");
  target.innerHTML = `<div class="loading">Loading…</div>`;
  let repo, tag;
  if (refStr.includes("@")) [repo, tag] = refStr.split("@", 2);
  else { repo = refStr; tag = "latest"; }
  if (!repo.includes("/")) {
    target.innerHTML = `<div class="error">Expected <code>user/repo</code> or <code>user/repo@tag</code>.</div>`;
    return;
  }
  try {
    const release = await fetchRelease(repo, tag);
    const tomlText = await fetchTomlAssetText(release);
    const recipe = parseTOML(tomlText);
    renderRecipe(repo, tag, release, recipe);
  } catch (e) {
    target.innerHTML = `<div class="error">${escapeHTML(e.message)}</div>`;
  }
}

// ---------- Search: GitHub code search for recipe.toml ----------
async function searchRecipes(filter) {
  const target = $("search-result");
  // Combine our marker (the recipe.toml literal) with optional user filter.
  // The format string `version = "0.1"` inside the recipe is too specific
  // to many TOML files; we anchor on the filename + the [base] section
  // being the second-most stable thing recipes have.
  let q = "filename:recipe.toml";
  if (filter && filter.trim()) q += " " + filter.trim();
  const url = `https://api.github.com/search/code?q=${encodeURIComponent(q)}&per_page=30`;

  target.innerHTML = `<div class="search-loading">Searching GitHub for <code>${escapeHTML(q)}</code>…</div>`;
  let res;
  try {
    res = await fetch(url, {
      headers: { Accept: "application/vnd.github+json" },
    });
  } catch (e) {
    target.innerHTML = `<div class="error">Network error: ${escapeHTML(e.message)}</div>`;
    return;
  }

  if (res.status === 422) {
    // Common for unauthenticated code search. Show what we asked for.
    target.innerHTML = `<div class="error">GitHub Search needs an authenticated request for code search. Sign in to GitHub in another tab, or use the
      <a href="https://github.com/search?q=${encodeURIComponent(q)}&type=code" target="_blank">GitHub UI</a> directly.</div>`;
    return;
  }
  if (res.status === 403) {
    target.innerHTML = `<div class="error">Search rate limit exceeded. Try again in a minute, or sign in to GitHub in another tab.</div>`;
    return;
  }
  if (!res.ok) {
    target.innerHTML = `<div class="error">GitHub API error: ${res.status}</div>`;
    return;
  }

  const data = await res.json();
  const items = data.items || [];
  if (items.length === 0) {
    target.innerHTML = `<div class="search-empty">No recipes match. Use the Publish tab to create the first one.</div>`;
    return;
  }

  // Each result is a code-search hit. We grab the raw file and parse a
  // small subset to summarize. Keep this fast: the response text is
  // typically a few hundred bytes.
  const out = [];
  for (const item of items.slice(0, 30)) {
    const repoFull = item.repository.full_name;
    const path = item.path;
    const rawUrl = `https://raw.githubusercontent.com/${repoFull}/HEAD/${path}`;
    let parsed;
    try {
      const t = await fetch(rawUrl);
      if (!t.ok) continue;
      parsed = parseTOML(await t.text());
    } catch {
      continue;
    }
    if (!parsed.recipe || !parsed.base) continue; // not actually a recipe
    const name = parsed.recipe.name || repoFull.split("/")[1];
    const baseRef = parsed.base.ref || "?";
    const adapter0 = (parsed.adapters || [])[0] || {};
    const cfg = [];
    if (adapter0.rank != null) cfg.push(`r${adapter0.rank}`);
    if (adapter0.type) cfg.push(adapter0.type);
    out.push({
      name,
      repo: repoFull,
      path,
      base: baseRef,
      cfg: cfg.join(" · "),
    });
  }

  if (out.length === 0) {
    target.innerHTML = `<div class="search-empty">Found ${items.length} files matching <code>recipe.toml</code>, but none parsed as an mlrecipe v0.1 recipe.</div>`;
    return;
  }

  const html = out.map((r) => `
    <div class="search-result">
      <div>
        <div class="search-name"><a href="?ref=${encodeURIComponent(r.repo + "@latest")}#browse">${escapeHTML(r.name)}</a></div>
        <div class="search-base">${escapeHTML(r.repo)} · base <code>${escapeHTML(r.base)}</code>${r.cfg ? " · " + escapeHTML(r.cfg) : ""}</div>
      </div>
      <div class="search-meta"><a href="https://github.com/${r.repo}" target="_blank">repo</a></div>
    </div>
  `).join("");
  target.innerHTML = html;
}

// ---------- Tabs ----------
function setupTabs() {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      const which = btn.dataset.tab;
      document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
      document.querySelectorAll(".pane").forEach((p) =>
        p.classList.toggle("active", p.id === `pane-${which}`)
      );
      // Reflect in URL hash without scrolling.
      history.replaceState(null, "", `#${which}`);
    });
  });
  // Honor initial hash.
  const hash = location.hash.replace("#", "");
  if (hash && ["browse", "search", "publish"].includes(hash)) {
    document.querySelector(`.tab[data-tab="${hash}"]`)?.click();
  }
}

// ---------- Copy buttons ----------
function setupCopyButtons() {
  document.querySelectorAll(".copy-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const text = btn.dataset.copy;
      try {
        await navigator.clipboard.writeText(text);
        const orig = btn.textContent;
        btn.textContent = "Copied";
        setTimeout(() => { btn.textContent = orig; }, 1200);
      } catch {
        // Older browsers: fall back to selection.
        const ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand("copy"); } catch {}
        ta.remove();
      }
    });
  });
}

// ---------- Bootstrap ----------
$("browse-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const ref = $("browse-ref").value.trim();
  if (ref) {
    history.replaceState(null, "", `?ref=${encodeURIComponent(ref)}#browse`);
    inspectRef(ref);
  }
});
document.querySelectorAll(".examples code").forEach((c) => {
  c.addEventListener("click", () => {
    const ref = c.dataset.ref;
    $("browse-ref").value = ref;
    history.replaceState(null, "", `?ref=${encodeURIComponent(ref)}#browse`);
    inspectRef(ref);
  });
});

$("search-form").addEventListener("submit", (e) => {
  e.preventDefault();
  searchRecipes($("search-q").value);
});

setupTabs();
setupCopyButtons();

// Auto-load on ?ref= URL.
const urlRef = new URLSearchParams(location.search).get("ref");
if (urlRef) {
  $("browse-ref").value = urlRef;
  // Make sure the Browse tab is showing.
  document.querySelector('.tab[data-tab="browse"]')?.click();
  inspectRef(urlRef);
}
