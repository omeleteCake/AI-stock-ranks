const dom = {
  fileInput: document.getElementById("fileInput"),
  loadDemoBtn: document.getElementById("loadDemoBtn"),
  searchInput: document.getElementById("searchInput"),
  minTop20Input: document.getElementById("minTop20Input"),
  minTop20Value: document.getElementById("minTop20Value"),
  maxStdInput: document.getElementById("maxStdInput"),
  maxStdValue: document.getElementById("maxStdValue"),
  showExcludedInput: document.getElementById("showExcludedInput"),
  tableBody: document.getElementById("tableBody"),
  rowCountLabel: document.getElementById("rowCountLabel"),
  scatterSvg: document.getElementById("scatterSvg"),
  topList: document.getElementById("topList"),
  cautionList: document.getElementById("cautionList"),
  kpiUniverse: document.getElementById("kpiUniverse"),
  kpiIncluded: document.getElementById("kpiIncluded"),
  kpiExcluded: document.getElementById("kpiExcluded"),
  kpiTop20: document.getElementById("kpiTop20"),
  kpiStd: document.getElementById("kpiStd"),
};

const state = {
  allRows: [],
  filteredRows: [],
};

function toNumber(value, fallback = NaN) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function toText(value) {
  if (value === null || value === undefined) {
    return "";
  }
  return String(value).trim();
}

function normalizeRow(row) {
  return {
    ticker: toText(row.ticker).toUpperCase(),
    mean_rank: toNumber(row.mean_rank),
    std_rank: toNumber(row.std_rank),
    top20_rate: toNumber(row.top20_rate, 0),
    notes: toText(row.notes),
    exclusion_reason: toText(row.exclusion_reason),
  };
}

function parseJsonText(text) {
  const payload = JSON.parse(text);
  if (Array.isArray(payload)) {
    return payload.map(normalizeRow);
  }
  if (payload && Array.isArray(payload.aggregate)) {
    return payload.aggregate.map(normalizeRow);
  }
  throw new Error("JSON must be an array of aggregate rows.");
}

function parseCsvLine(line) {
  const out = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        cur += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }
    if (ch === "," && !inQuotes) {
      out.push(cur);
      cur = "";
      continue;
    }
    cur += ch;
  }
  out.push(cur);
  return out.map((x) => x.trim());
}

function parseCsvText(text) {
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (lines.length < 2) {
    throw new Error("CSV must include a header and at least one data row.");
  }
  const headers = parseCsvLine(lines[0]);
  const rows = [];
  for (let i = 1; i < lines.length; i += 1) {
    const cells = parseCsvLine(lines[i]);
    const row = {};
    headers.forEach((h, idx) => {
      row[h] = idx < cells.length ? cells[idx] : "";
    });
    rows.push(normalizeRow(row));
  }
  return rows;
}

function statusForRow(row) {
  if (row.exclusion_reason) {
    return { text: "Excluded", cls: "status-excluded" };
  }
  if (row.top20_rate >= 0.8 && row.std_rank <= 1.4) {
    return { text: "Conviction", cls: "status-conviction" };
  }
  return { text: "Watch", cls: "status-watch" };
}

function median(values) {
  if (!values.length) {
    return NaN;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) {
    return sorted[mid];
  }
  return (sorted[mid - 1] + sorted[mid]) / 2;
}

function applyFilters() {
  const search = dom.searchInput.value.trim().toUpperCase();
  const minTop20 = Number(dom.minTop20Input.value);
  const maxStd = Number(dom.maxStdInput.value);
  const showExcluded = dom.showExcludedInput.checked;

  const ranked = [...state.allRows].sort((a, b) => a.mean_rank - b.mean_rank);
  state.filteredRows = ranked.filter((row) => {
    if (search && !row.ticker.includes(search)) {
      return false;
    }
    if (row.top20_rate < minTop20) {
      return false;
    }
    if (Number.isFinite(row.std_rank) && row.std_rank > maxStd) {
      return false;
    }
    if (!showExcluded && row.exclusion_reason) {
      return false;
    }
    return true;
  });

  dom.minTop20Value.textContent = minTop20.toFixed(2);
  dom.maxStdValue.textContent = maxStd.toFixed(1);

  renderKpis();
  renderTable();
  renderInsights();
  renderScatter();
}

function renderKpis() {
  const rows = state.filteredRows;
  const included = rows.filter((r) => !r.exclusion_reason);
  const excluded = rows.filter((r) => !!r.exclusion_reason);
  const avgTop20 =
    rows.length > 0
      ? rows.reduce((acc, r) => acc + (Number.isFinite(r.top20_rate) ? r.top20_rate : 0), 0) /
        rows.length
      : 0;
  const medStd = median(rows.map((r) => r.std_rank).filter((x) => Number.isFinite(x)));

  dom.kpiUniverse.textContent = String(rows.length);
  dom.kpiIncluded.textContent = String(included.length);
  dom.kpiExcluded.textContent = String(excluded.length);
  dom.kpiTop20.textContent = `${(avgTop20 * 100).toFixed(1)}%`;
  dom.kpiStd.textContent = Number.isFinite(medStd) ? medStd.toFixed(2) : "-";
}

function renderTable() {
  const rows = state.filteredRows;
  dom.tableBody.innerHTML = "";
  rows.forEach((row, idx) => {
    const status = statusForRow(row);
    const tr = document.createElement("tr");
    const notes = row.notes || row.exclusion_reason || "";
    tr.innerHTML = `
      <td>${idx + 1}</td>
      <td><strong>${row.ticker}</strong></td>
      <td>${Number.isFinite(row.mean_rank) ? row.mean_rank.toFixed(2) : "-"}</td>
      <td>${Number.isFinite(row.std_rank) ? row.std_rank.toFixed(2) : "-"}</td>
      <td>${Number.isFinite(row.top20_rate) ? (row.top20_rate * 100).toFixed(1) + "%" : "-"}</td>
      <td><span class="status-chip ${status.cls}">${status.text}</span></td>
      <td class="note-cell">${escapeHtml(notes).slice(0, 220)}</td>
    `;
    dom.tableBody.appendChild(tr);
  });
  dom.rowCountLabel.textContent = `${rows.length} rows`;
}

function makeListItem(row, scoreText, noteText) {
  const li = document.createElement("li");
  li.className = "list-item";
  li.innerHTML = `
    <div><strong>${row.ticker}</strong> <span>${scoreText}</span></div>
    <div class="list-note">${escapeHtml(noteText)}</div>
  `;
  return li;
}

function renderInsights() {
  dom.topList.innerHTML = "";
  dom.cautionList.innerHTML = "";

  const included = state.filteredRows
    .filter((r) => !r.exclusion_reason)
    .sort((a, b) => a.mean_rank - b.mean_rank)
    .slice(0, 6);
  included.forEach((row) => {
    dom.topList.appendChild(
      makeListItem(
        row,
        `rank ${row.mean_rank.toFixed(2)} | top20 ${(row.top20_rate * 100).toFixed(1)}%`,
        row.notes || "No additional notes."
      )
    );
  });

  const caution = state.filteredRows
    .filter((r) => r.exclusion_reason || /RED FLAG/i.test(r.notes))
    .sort((a, b) => (a.exclusion_reason ? -1 : 0) - (b.exclusion_reason ? -1 : 0))
    .slice(0, 6);

  caution.forEach((row) => {
    const reason = row.exclusion_reason || "flagged_note";
    dom.cautionList.appendChild(
      makeListItem(row, reason, row.notes || "No additional notes.")
    );
  });

  if (!included.length) {
    dom.topList.appendChild(makeListItem({ ticker: "-" }, "", "No included rows."));
  }
  if (!caution.length) {
    dom.cautionList.appendChild(makeListItem({ ticker: "-" }, "", "No caution rows."));
  }
}

function makeSvgNode(name, attrs) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.keys(attrs).forEach((k) => node.setAttribute(k, String(attrs[k])));
  return node;
}

function renderScatter() {
  const svg = dom.scatterSvg;
  svg.innerHTML = "";
  const rows = state.filteredRows.filter(
    (r) => Number.isFinite(r.mean_rank) && Number.isFinite(r.std_rank)
  );
  if (!rows.length) {
    return;
  }

  const width = 640;
  const height = 360;
  const padL = 56;
  const padR = 22;
  const padT = 24;
  const padB = 42;
  const maxX = Math.max(...rows.map((r) => r.std_rank), 1);
  const maxY = Math.max(...rows.map((r) => r.mean_rank), 1);

  const toX = (x) => padL + (x / maxX) * (width - padL - padR);
  const toY = (y) => padT + ((y - 1) / Math.max(maxY - 1, 1e-6)) * (height - padT - padB);

  svg.appendChild(
    makeSvgNode("line", {
      x1: padL,
      y1: height - padB,
      x2: width - padR,
      y2: height - padB,
      stroke: "#9aa8a0",
      "stroke-width": 1,
    })
  );
  svg.appendChild(
    makeSvgNode("line", {
      x1: padL,
      y1: padT,
      x2: padL,
      y2: height - padB,
      stroke: "#9aa8a0",
      "stroke-width": 1,
    })
  );

  const xLabel = makeSvgNode("text", {
    x: width / 2,
    y: height - 10,
    "text-anchor": "middle",
    fill: "#4c5d56",
    "font-size": "12",
  });
  xLabel.textContent = "Std Rank (lower is steadier)";
  svg.appendChild(xLabel);

  const yLabel = makeSvgNode("text", {
    x: 16,
    y: height / 2,
    transform: `rotate(-90 16 ${height / 2})`,
    "text-anchor": "middle",
    fill: "#4c5d56",
    "font-size": "12",
  });
  yLabel.textContent = "Mean Rank (lower is better)";
  svg.appendChild(yLabel);

  rows.forEach((row) => {
    const cx = toX(row.std_rank);
    const cy = toY(row.mean_rank);
    const excluded = !!row.exclusion_reason;
    const circle = makeSvgNode("circle", {
      cx,
      cy,
      r: excluded ? 4 : 5.2,
      fill: excluded ? "#dd4f39" : "#0a7f7a",
      opacity: excluded ? 0.75 : 0.86,
    });
    const tip = makeSvgNode("title", {});
    tip.textContent = `${row.ticker} | mean ${row.mean_rank.toFixed(2)} | std ${row.std_rank.toFixed(
      2
    )} | top20 ${(row.top20_rate * 100).toFixed(1)}%`;
    circle.appendChild(tip);
    svg.appendChild(circle);
  });
}

function escapeHtml(s) {
  return s
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

async function loadDemo() {
  const resp = await fetch("./demo_assessment.json");
  if (!resp.ok) {
    throw new Error("Could not load demo_assessment.json");
  }
  const payload = await resp.json();
  state.allRows = payload.map(normalizeRow).sort((a, b) => a.mean_rank - b.mean_rank);
  const maxStd = Math.max(...state.allRows.map((r) => (Number.isFinite(r.std_rank) ? r.std_rank : 0)), 10);
  dom.maxStdInput.value = String(Math.ceil(maxStd * 10) / 10);
  applyFilters();
}

async function loadFromFile(file) {
  const text = await file.text();
  const lower = file.name.toLowerCase();
  const rows = lower.endsWith(".csv") ? parseCsvText(text) : parseJsonText(text);
  state.allRows = rows
    .filter((r) => r.ticker)
    .sort((a, b) => a.mean_rank - b.mean_rank);

  const maxStd = Math.max(...state.allRows.map((r) => (Number.isFinite(r.std_rank) ? r.std_rank : 0)), 10);
  dom.maxStdInput.value = String(Math.ceil(maxStd * 10) / 10);
  applyFilters();
}

function bindEvents() {
  dom.fileInput.addEventListener("change", async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) {
      return;
    }
    try {
      await loadFromFile(file);
    } catch (err) {
      alert(`Could not parse file: ${err.message}`);
    }
  });
  dom.loadDemoBtn.addEventListener("click", async () => {
    try {
      await loadDemo();
    } catch (err) {
      alert(err.message);
    }
  });
  dom.searchInput.addEventListener("input", applyFilters);
  dom.minTop20Input.addEventListener("input", applyFilters);
  dom.maxStdInput.addEventListener("input", applyFilters);
  dom.showExcludedInput.addEventListener("change", applyFilters);
}

async function init() {
  bindEvents();
  try {
    await loadDemo();
  } catch (err) {
    console.error(err);
  }
}

init();
