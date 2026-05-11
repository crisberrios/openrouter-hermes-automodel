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

function fmtReleased(isoDate, ageDays) {
  if (!isoDate) return { text: "—", title: "release date unknown", unknown: true };
  let label = isoDate;
  if (Number.isFinite(ageDays)) {
    if (ageDays < 1) label = "today";
    else if (ageDays < 7) label = `${ageDays}d ago`;
    else if (ageDays < 60) label = `${Math.round(ageDays / 7)}w ago`;
    else if (ageDays < 730) label = `${Math.round(ageDays / 30)}mo ago`;
    else label = `${Math.round(ageDays / 365)}y ago`;
  }
  return { text: label, title: `${isoDate} (${ageDays}d)`, unknown: false };
}

function renderRow(m, kind) {
  const tr = document.createElement("tr");
  const promptPrice = m.pricing?.prompt_usd_per_mtok;
  const completionPrice = m.pricing?.completion_usd_per_mtok;
  const rel = fmtReleased(m.release_date, m.age_days);
  const valueCell = kind === "free"
    ? ""
    : `<td class="num" data-label="value">${fmtScore(m.scores?.value_score)}</td>`;

  tr.innerHTML = `
    <td class="num rank" data-label="rank">${m.rank}</td>
    <td class="id" data-label="id">
      <code>${escapeHtml(m.id)}</code>${m.is_free ? '<span class="free-tag">free</span>' : ""}
      <span class="name">${escapeHtml(m.name || "")}</span>
    </td>
    <td class="num" data-label="ctx">${fmtCtx(m.context_length)}</td>
    <td class="released${rel.unknown ? " unknown" : ""}" data-label="released" title="${escapeHtml(rel.title)}">${escapeHtml(rel.text)}</td>
    <td class="${m.supports_reasoning ? "flag-yes" : "flag-no"}" data-label="reason">${m.supports_reasoning ? "✓" : "·"}</td>
    <td class="num" data-label="quality">${fmtScore(m.scores?.quality_score)}</td>
    ${valueCell}
    <td class="num ${promptPrice === 0 ? "price-free" : "price"}" data-label="$/M in">${fmtPrice(promptPrice)}</td>
    <td class="num ${completionPrice === 0 ? "price-free" : "price"}" data-label="$/M out">${fmtPrice(completionPrice)}</td>
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
  for (const m of data.models || []) tbody.appendChild(renderRow(m, kind));
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

function wireQualityExplainers() {
  const headers = document.querySelectorAll("th.th-explain");
  headers.forEach(th => {
    th.addEventListener("click", e => {
      // Toggle pinned state. Click anywhere else dismisses.
      e.stopPropagation();
      const wasPinned = th.classList.contains("pinned");
      headers.forEach(h => h.classList.remove("pinned"));
      if (!wasPinned) th.classList.add("pinned");
    });
    th.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        th.click();
      } else if (e.key === "Escape") {
        th.classList.remove("pinned");
      }
    });
  });
  document.addEventListener("click", () => {
    headers.forEach(h => h.classList.remove("pinned"));
  });
}

(async function main() {
  const results = await Promise.all(LISTS.map(loadOne));
  const first = results.find(Boolean);
  renderHeader(first);
  wireQualityExplainers();
})();
