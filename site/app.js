// Dev demo for the automodel JSON outputs.
// Fetches /free.json, /balanced.json, /best.json from the same origin
// and renders each into the matching section[data-kind=...].

const LISTS = ["free", "balanced", "best"];

function fmtCtx(n) {
  if (!n) return "—";
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(n % 1_000_000 ? 2 : 0) + "M";
  if (n >= 1_000) return Math.round(n / 1_000) + "k";
  return String(n);
}

function fmtPrice(p) {
  if (p == null) return "—";
  if (p === 0) return "0";
  if (p < 0.01) return p.toFixed(4);
  if (p < 1) return p.toFixed(3);
  return p.toFixed(2);
}

function fmtScore(s) {
  if (s == null) return "—";
  return Number(s).toFixed(3);
}

function fmtTimestamp(iso) {
  if (!iso) return "unknown";
  try {
    const d = new Date(iso);
    return d.toISOString().replace("T", " ").replace(/\.\d+Z$/, "Z");
  } catch {
    return iso;
  }
}

function renderRow(m) {
  const tr = document.createElement("tr");
  const promptPrice = m.pricing?.prompt_usd_per_mtok;
  const completionPrice = m.pricing?.completion_usd_per_mtok;

  tr.innerHTML = `
    <td class="num rank">${m.rank}</td>
    <td class="id">
      <code>${escapeHtml(m.id)}</code>${m.is_free ? '<span class="free-tag">free</span>' : ""}
      <span class="name">${escapeHtml(m.name || "")}</span>
    </td>
    <td class="num">${fmtCtx(m.context_length)}</td>
    <td class="${m.supports_tools ? "flag-yes" : "flag-no"}">${m.supports_tools ? "✓" : "·"}</td>
    <td class="${m.supports_reasoning ? "flag-yes" : "flag-no"}">${m.supports_reasoning ? "✓" : "·"}</td>
    <td class="num">${fmtScore(m.scores?.quality_score)}</td>
    <td class="num">${fmtScore(m.scores?.value_score)}</td>
    <td class="num ${promptPrice === 0 ? "price-free" : "price"}">${fmtPrice(promptPrice)}</td>
    <td class="num ${completionPrice === 0 ? "price-free" : "price"}">${fmtPrice(completionPrice)}</td>
  `;
  return tr;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

function renderError(section, msg) {
  const wrap = section.querySelector(".table-wrap");
  const err = document.createElement("div");
  err.className = "error";
  err.textContent = msg;
  wrap.replaceWith(err);
}

async function loadOne(kind) {
  const section = document.querySelector(`section.list[data-kind="${kind}"]`);
  if (!section) return null;

  let data;
  try {
    const res = await fetch(`./${kind}.json`, { cache: "no-cache" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
  } catch (e) {
    renderError(section, `Failed to load ${kind}.json: ${e.message}`);
    return null;
  }

  const tbody = section.querySelector("tbody");
  tbody.innerHTML = "";
  for (const m of data.models || []) tbody.appendChild(renderRow(m));
  return data;
}

function renderHeader(first) {
  if (!first) return;
  const gen = document.getElementById("generated-at");
  if (gen) gen.textContent = fmtTimestamp(first.generated_at);

  const apps = document.getElementById("tracked-apps");
  if (apps && first.tracked_apps) {
    const parts = Object.entries(first.tracked_apps)
      .sort((a, b) => (a[1].rank ?? 9999) - (b[1].rank ?? 9999))
      .map(([slug, info]) => `${slug}#${info.rank ?? "?"}`);
    apps.textContent = parts.join(", ");
  }
}

(async function main() {
  const results = await Promise.all(LISTS.map(loadOne));
  const first = results.find(Boolean);
  renderHeader(first);
})();
