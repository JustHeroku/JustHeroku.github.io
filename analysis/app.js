const state = {
  runs: [],
  run: null,
  history: [],
  confusion: null,
  classData: [],
  filterMaxAcc: 100,
  graphMinConfusions: 0,
  graphShowAllEdges: false,
  selectedClasses: new Set(),
  activeTab: "history",
};

const elements = {
  runButton: document.getElementById("run-button"),
  runMenu: document.getElementById("run-menu"),
  currentRun: document.getElementById("current-run"),
  status: document.getElementById("status"),
  tooltip: document.getElementById("tooltip"),
  filterRow: document.getElementById("filter-row"),
  filterInput: document.getElementById("accuracy-filter"),
  filterValue: document.getElementById("filter-value"),
  graphFilterRow: document.getElementById("graph-filter-row"),
  graphFilterInput: document.getElementById("graph-filter"),
  graphFilterValue: document.getElementById("graph-filter-value"),
  graphEdgeToggle: document.getElementById("graph-edge-toggle"),
  avgAll: document.getElementById("avg-all"),
  avgAllCount: document.getElementById("avg-all-count"),
  avgSelected: document.getElementById("avg-selected"),
  selectedCount: document.getElementById("selected-count"),
  selectVisible: document.getElementById("select-visible"),
  selectAll: document.getElementById("select-all"),
  clearSelection: document.getElementById("clear-selection"),
  historyMeta: document.getElementById("history-meta"),
  perClassMeta: document.getElementById("per-class-meta"),
  matrixMeta: document.getElementById("matrix-meta"),
  graphMeta: document.getElementById("graph-meta"),
};

const rootStyles = getComputedStyle(document.documentElement);
const colors = {
  train: rootStyles.getPropertyValue("--accent").trim() || "#e76f51",
  val: rootStyles.getPropertyValue("--accent-2").trim() || "#2a9d8f",
  correct: rootStyles.getPropertyValue("--accent-strong").trim() || "#c65236",
  grid: rootStyles.getPropertyValue("--line").trim() || "#e2d7c6",
};

function setStatus(message) {
  elements.status.textContent = message;
}

function shortLabel(full) {
  if (!full) {
    return "";
  }
  const lastIdx = full.lastIndexOf("__");
  const suffix = lastIdx >= 0 ? full.slice(lastIdx + 2) : full;
  const first = full[0] || "";
  return `${first}_${suffix}`;
}

function showTooltip(html, event) {
  elements.tooltip.innerHTML = html;
  elements.tooltip.classList.remove("hidden");
  elements.tooltip.style.left = `${event.clientX + 12}px`;
  elements.tooltip.style.top = `${event.clientY + 12}px`;
}

function hideTooltip() {
  elements.tooltip.classList.add("hidden");
}

async function loadRuns() {
  try {
    const res = await fetch("./runs.json");
    const data = await res.json();
    if (Array.isArray(data) && data.length > 0) {
      state.runs = data;
    }
  } catch (err) {
    state.runs = [];
  }

  if (state.runs.length === 0) {
    state.runs = ["conv_max_256_final"];
  }

  renderRunMenu();
  await loadRun(state.runs[0]);
}

function renderRunMenu() {
  elements.runMenu.innerHTML = "";
  state.runs.forEach((runName) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = runName;
    btn.addEventListener("click", async () => {
      elements.runMenu.classList.add("hidden");
      await loadRun(runName);
    });
    elements.runMenu.appendChild(btn);
  });
}

function parseHistory(text) {
  const rows = d3.csvParse(text);
  return rows
    .map((row) => ({
      epoch: Number(row.epoch),
      accuracy: Number(row.accuracy),
      val_accuracy: Number(row.val_accuracy),
      loss: Number(row.loss),
      val_loss: Number(row.val_loss),
    }))
    .filter((row) => Number.isFinite(row.epoch));
}

function parseConfusion(text) {
  const rows = d3.csvParse(text);
  const labels = rows.columns.slice(1);
  const matrix = rows.map((row) => labels.map((label) => Number(row[label])));
  return { labels, matrix };
}

function buildClassData(confusion) {
  const totals = confusion.matrix.map((row) => row.reduce((a, b) => a + b, 0));
  const classData = confusion.labels.map((label, idx) => {
    const total = totals[idx];
    const correct = confusion.matrix[idx][idx] || 0;
    const acc = total > 0 ? correct / total : 0;
    const rowPct = confusion.matrix[idx].map((val) => (total > 0 ? val / total : 0));
    return {
      index: idx,
      full: label,
      short: shortLabel(label),
      total,
      correct,
      acc,
      row: confusion.matrix[idx],
      rowPct,
    };
  });
  return classData;
}

function filteredClasses() {
  if (state.filterMaxAcc >= 100) {
    return state.classData;
  }
  return state.classData.filter((d) => d.acc * 100 <= state.filterMaxAcc + 1e-9);
}

function computeAverage(indices) {
  let sum = 0;
  let count = 0;
  indices.forEach((idx) => {
    const item = state.classData[idx];
    if (item && item.total > 0 && Number.isFinite(item.acc)) {
      sum += item.acc;
      count += 1;
    }
  });
  return { avg: count ? sum / count : null, count };
}

function updatePerClassSummary() {
  if (!elements.avgAll || !elements.avgSelected) {
    return;
  }
  const allIndices = state.classData.map((d) => d.index);
  const allStats = computeAverage(allIndices);
  const selectedIndices = Array.from(state.selectedClasses);
  const selectedStats = computeAverage(selectedIndices);

  elements.avgAll.textContent =
    allStats.avg !== null ? `${(allStats.avg * 100).toFixed(2)}%` : "n/a";
  elements.avgAllCount.textContent = `${allStats.count} classes`;

  elements.avgSelected.textContent =
    selectedStats.avg !== null ? `${(selectedStats.avg * 100).toFixed(2)}%` : "n/a";
  elements.selectedCount.textContent = `${selectedIndices.length} selected`;
}

function toggleSelection(idx) {
  if (state.selectedClasses.has(idx)) {
    state.selectedClasses.delete(idx);
  } else {
    state.selectedClasses.add(idx);
  }
  hideTooltip();
  renderPerClass();
}

function setSelection(indices) {
  state.selectedClasses = new Set(indices);
  renderPerClass();
}

function graphCandidateIndices() {
  const cm = state.confusion.matrix;
  if (state.graphMinConfusions <= 0) {
    return state.classData.map((d) => d.index);
  }
  return state.classData
    .filter((d) => {
      let maxConf = 0;
      for (let j = 0; j < cm.length; j += 1) {
        if (j === d.index) {
          continue;
        }
        const pairMax = Math.max(cm[d.index][j], cm[j][d.index]);
        if (pairMax > maxConf) {
          maxConf = pairMax;
        }
      }
      return maxConf >= state.graphMinConfusions;
    })
    .map((d) => d.index);
}

async function loadRun(runName) {
  setStatus(`Loading ${runName}...`);
  elements.currentRun.textContent = runName;
  state.run = runName;

  try {
    const [historyText, confusionText] = await Promise.all([
      fetch(`./${runName}/history.csv`).then((r) => r.text()),
      fetch(`./${runName}/confusion_matrix.csv`).then((r) => r.text()),
    ]);
    state.history = parseHistory(historyText);
    state.confusion = parseConfusion(confusionText);
    state.classData = buildClassData(state.confusion);
    state.selectedClasses = new Set();
    elements.filterValue.textContent = String(state.filterMaxAcc);
    elements.graphFilterValue.textContent = String(state.graphMinConfusions);
    if (elements.graphEdgeToggle) {
      elements.graphEdgeToggle.checked = state.graphShowAllEdges;
    }
    renderActiveTab();
    setStatus(`Loaded ${runName}`);
  } catch (err) {
    setStatus(`Failed to load ${runName}.`);
  }
}

function renderActiveTab() {
  if (!state.confusion) {
    return;
  }
  if (state.activeTab === "history") {
    renderHistory();
  } else if (state.activeTab === "per-class") {
    renderPerClass();
  } else if (state.activeTab === "matrix") {
    renderConfusionMatrix();
  } else if (state.activeTab === "graph") {
    renderConfusionGraph();
  }
}

function renderHistory() {
  const history = state.history;
  if (!history.length) {
    elements.historyMeta.textContent = "No history data";
    return;
  }
  const trainAcc = history.map((d) => d.accuracy).filter((v) => Number.isFinite(v));
  const valAcc = history.map((d) => d.val_accuracy).filter((v) => Number.isFinite(v));
  const trainLoss = history.map((d) => d.loss).filter((v) => Number.isFinite(v));
  const valLoss = history.map((d) => d.val_loss).filter((v) => Number.isFinite(v));

  const maxTrainAcc = d3.max(trainAcc);
  const maxValAcc = d3.max(valAcc);
  const minTrainLoss = d3.min(trainLoss);
  const minValLoss = d3.min(valLoss);

  const fmtPct = (v) => (Number.isFinite(v) ? `${(v * 100).toFixed(2)}%` : "n/a");
  const fmtLoss = (v) => (Number.isFinite(v) ? v.toFixed(4) : "n/a");

  elements.historyMeta.textContent =
    `Epochs: ${history.length} | Max acc: train ${fmtPct(maxTrainAcc)} ` +
    `val ${fmtPct(maxValAcc)} | Min loss: train ${fmtLoss(minTrainLoss)} ` +
    `val ${fmtLoss(minValLoss)}`;

  const accSeries = [
    {
      name: "train",
      color: colors.train,
      values: history.map((d) => ({ x: d.epoch + 1, y: d.accuracy })),
    },
    {
      name: "val",
      color: colors.val,
      values: history.map((d) => ({ x: d.epoch + 1, y: d.val_accuracy })),
    },
  ];

  const lossSeries = [
    {
      name: "train",
      color: colors.train,
      values: history.map((d) => ({ x: d.epoch + 1, y: d.loss })),
    },
    {
      name: "val",
      color: colors.val,
      values: history.map((d) => ({ x: d.epoch + 1, y: d.val_loss })),
    },
  ];

  renderLineChart("history-accuracy", accSeries, {
    yDomain: [0, 1],
    yLabel: "Accuracy",
  });
  const maxLoss = d3.max(lossSeries.flatMap((s) => s.values.map((v) => v.y)));
  const lossMax = Number.isFinite(maxLoss) ? maxLoss * 1.05 : 1;
  renderLineChart("history-loss", lossSeries, {
    yDomain: [0, lossMax],
    yLabel: "Loss",
  });
}

function renderLineChart(containerId, series, options) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  const width = container.clientWidth || 640;
  const height = 260;
  const margin = { top: 20, right: 24, bottom: 34, left: 48 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  const x = d3
    .scaleLinear()
    .domain(d3.extent(series[0].values, (d) => d.x))
    .range([0, innerWidth]);

  const y = d3
    .scaleLinear()
    .domain(options.yDomain)
    .nice()
    .range([innerHeight, 0]);

  const svg = d3
    .select(container)
    .append("svg")
    .attr("width", width)
    .attr("height", height);

  const chart = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  chart
    .append("g")
    .call(d3.axisLeft(y).ticks(5).tickSize(-innerWidth).tickFormat(""))
    .call((g) => g.selectAll("line").attr("stroke", colors.grid))
    .call((g) => g.selectAll("path").remove());

  chart.append("g").call(d3.axisLeft(y).ticks(5));
  chart
    .append("g")
    .attr("transform", `translate(0,${innerHeight})`)
    .call(d3.axisBottom(x).ticks(6));

  const line = d3
    .line()
    .x((d) => x(d.x))
    .y((d) => y(d.y));

  series.forEach((s) => {
    chart
      .append("path")
      .datum(s.values)
      .attr("fill", "none")
      .attr("stroke", s.color)
      .attr("stroke-width", 2.4)
      .attr("d", line);
  });

  const legend = svg.append("g").attr("transform", `translate(${margin.left},8)`);
  series.forEach((s, idx) => {
    const group = legend.append("g").attr("transform", `translate(${idx * 90},0)`);
    group
      .append("rect")
      .attr("width", 12)
      .attr("height", 12)
      .attr("fill", s.color)
      .attr("rx", 3);
    group
      .append("text")
      .attr("x", 18)
      .attr("y", 10)
      .attr("font-size", 12)
      .text(s.name);
  });
}

function renderPerClass() {
  const scrollX = window.scrollX;
  const scrollY = window.scrollY;
  const data = filteredClasses();
  const totalClasses = state.classData.length;
  elements.perClassMeta.textContent = `Showing ${data.length} of ${totalClasses} classes`;

  const container = document.getElementById("per-class-chart");
  container.innerHTML = "";
  if (!data.length) {
    container.textContent = "No classes match this filter.";
    updatePerClassSummary();
    requestAnimationFrame(() => {
      window.scrollTo(scrollX, scrollY);
    });
    return;
  }

  const width = container.clientWidth || 860;
  const labelWidth = 90;
  const valueWidth = 70;
  const barHeight = 16;
  const gap = 10;
  const margin = { top: 10, bottom: 0 };
  const chartWidth = Math.max(40, width - labelWidth - valueWidth);
  const totalGap = data.length > 1 ? (data.length - 1) * gap : 0;
  const height = margin.top + margin.bottom + data.length * barHeight + totalGap;

  const svg = d3
    .select(container)
    .append("svg")
    .attr("width", width)
    .attr("height", height);

  const colorScale = d3.scaleSequential(d3.interpolateYlGnBu).domain([0, totalClasses - 1]);
  const lighten = (color) => d3.interpolateRgb(color, "#ffffff")(0.45);

  const rowGroup = svg
    .append("g")
    .selectAll("g")
    .data(data)
    .enter()
    .append("g")
    .attr("transform", (d, idx) => `translate(0, ${margin.top + idx * (barHeight + gap)})`)
    .style("cursor", "pointer")
    .on("click", (event, d) => {
      event.preventDefault();
      toggleSelection(d.index);
    });

  rowGroup.each(function (d) {
    const group = d3.select(this);
    if (state.selectedClasses.has(d.index)) {
      group
        .append("rect")
        .attr("x", 0)
        .attr("y", -2)
        .attr("width", width)
        .attr("height", barHeight + 4)
        .attr("fill", "rgba(42, 157, 143, 0.12)");
    }

    group
      .append("text")
      .attr("x", 0)
      .attr("y", barHeight - 2)
      .attr("font-size", 11)
      .text(d.short);

    const total = d.total;
    const baseX = labelWidth;
    group
      .append("rect")
      .attr("x", baseX)
      .attr("y", 0)
      .attr("width", chartWidth)
      .attr("height", barHeight)
      .attr("fill", "#f1ede4");

    const misSegments = [];
    d.row.forEach((count, j) => {
      if (j === d.index || count <= 0) {
        return;
      }
      misSegments.push({ predIndex: j, count, type: "mis" });
    });
    misSegments.sort((a, b) => b.count - a.count);
    const segments = [{ predIndex: d.index, count: d.correct, type: "correct" }].concat(
      misSegments
    );

    let cursor = baseX;
    segments.forEach((seg) => {
      const pct = total > 0 ? seg.count / total : 0;
      const w = chartWidth * pct;
      if (w <= 0) {
        return;
      }
      const baseColor = seg.type === "correct" ? colors.correct : colorScale(seg.predIndex);
      const fill = seg.type === "correct" ? baseColor : lighten(baseColor);
      group
        .append("rect")
        .attr("x", cursor)
        .attr("y", 0)
        .attr("width", w)
        .attr("height", barHeight)
        .attr("fill", fill)
        .attr("stroke", "#ffffff")
        .attr("stroke-width", 0.4)
        .on("mousemove", (event) => {
          const predShort = state.classData[seg.predIndex].short;
          const percent = total > 0 ? ((seg.count / total) * 100).toFixed(1) : "0.0";
          showTooltip(
            `True: ${d.short}<br>Pred: ${predShort}<br>Count: ${seg.count}<br>Percent: ${percent}%`,
            event
          );
        })
        .on("mouseleave", hideTooltip);
      cursor += w;
    });

    group
      .append("text")
      .attr("x", labelWidth + chartWidth + 8)
      .attr("y", barHeight - 2)
      .attr("font-size", 11)
      .attr("fill", colors.train)
      .text(`${(d.acc * 100).toFixed(1)}%`);
  });
  updatePerClassSummary();
  requestAnimationFrame(() => {
    window.scrollTo(scrollX, scrollY);
  });
}

function renderConfusionMatrix() {
  const data = filteredClasses();
  const totalClasses = state.classData.length;
  elements.matrixMeta.textContent = `Showing ${data.length} of ${totalClasses} classes`;

  const container = document.getElementById("confusion-matrix");
  container.innerHTML = "";
  if (!data.length) {
    container.textContent = "No classes match this filter.";
    return;
  }

  const indices = data.map((d) => d.index);
  const labels = data.map((d) => d.short);
  const matrix = indices.map((i) => indices.map((j) => state.confusion.matrix[i][j]));
  const rowTotals = indices.map((i) => state.classData[i].total);

  const cellSize = 18;
  const labelMargin = 80;
  const topMargin = 70;
  const width = labelMargin + cellSize * labels.length + 20;
  const height = topMargin + cellSize * labels.length + 20;

  const svg = d3
    .select(container)
    .append("svg")
    .attr("width", width)
    .attr("height", height);

  const color = d3.scaleSequential(d3.interpolateBlues).domain([0, 1]);

  labels.forEach((label, idx) => {
    svg
      .append("text")
      .attr("x", labelMargin + idx * cellSize + 4)
      .attr("y", topMargin - 10)
      .attr("font-size", 10)
      .attr("transform", `rotate(-45, ${labelMargin + idx * cellSize + 4}, ${topMargin - 10})`)
      .text(label);
  });

  labels.forEach((label, idx) => {
    svg
      .append("text")
      .attr("x", 10)
      .attr("y", topMargin + idx * cellSize + 12)
      .attr("font-size", 10)
      .text(label);
  });

  matrix.forEach((row, i) => {
    row.forEach((val, j) => {
      const pct = rowTotals[i] > 0 ? val / rowTotals[i] : 0;
      svg
        .append("rect")
        .attr("x", labelMargin + j * cellSize)
        .attr("y", topMargin + i * cellSize)
        .attr("width", cellSize)
        .attr("height", cellSize)
        .attr("fill", color(pct))
        .attr("stroke", "#ffffff")
        .attr("stroke-width", 0.5)
        .on("mousemove", (event) => {
          const percent = (pct * 100).toFixed(2);
          showTooltip(
            `True: ${labels[i]}<br>Pred: ${labels[j]}<br>Count: ${val}<br>Row pct: ${percent}%`,
            event
          );
        })
        .on("mouseleave", hideTooltip);
    });
  });
}

function renderConfusionGraph() {
  const totalClasses = state.classData.length;
  const container = document.getElementById("confusion-graph");
  container.innerHTML = "";
  const candidateIndices = graphCandidateIndices();

  if (!candidateIndices.length) {
    container.textContent = "No classes match this filter.";
    return;
  }

  const indices = candidateIndices.slice();
  const totals = state.classData.map((d) => d.total);
  const cm = state.confusion.matrix;

  if (indices.length < 2) {
    container.textContent = "Not enough classes to draw a graph.";
    return;
  }

  const scoreBetween = (a, b) => cm[a][b] + cm[b][a];
  const edges = [];
  const showAllEdges = state.graphShowAllEdges;
  const minEdge = showAllEdges ? 1 : 2;
  for (let a = 0; a < indices.length; a += 1) {
    for (let b = a + 1; b < indices.length; b += 1) {
      const i = indices[a];
      const j = indices[b];
      const pairMax = Math.max(cm[i][j], cm[j][i]);
      if (pairMax < minEdge) {
        continue;
      }
      const rateIJ = totals[i] > 0 ? cm[i][j] / totals[i] : 0;
      const rateJI = totals[j] > 0 ? cm[j][i] / totals[j] : 0;
      const avgRate = (rateIJ + rateJI) / 2;
      const countPair = cm[i][j] + cm[j][i];
      if (countPair <= 0) {
        continue;
      }
      edges.push({
        source: i,
        target: j,
        rateIJ,
        rateJI,
        avgRate,
        countIJ: cm[i][j],
        countJI: cm[j][i],
      });
      edges.push({
        source: j,
        target: i,
        rateIJ: rateJI,
        rateJI: rateIJ,
        avgRate,
        countIJ: cm[j][i],
        countJI: cm[i][j],
      });
    }
  }

  if (!edges.length) {
    container.textContent = "No confusions for these classes.";
    return;
  }
  const nodeIdSet = new Set();
  edges.forEach((edge) => {
    nodeIdSet.add(edge.source);
    nodeIdSet.add(edge.target);
  });
  const nodeIds = Array.from(nodeIdSet);
  if (nodeIds.length < 2) {
    container.textContent = "No connected classes for this filter.";
    return;
  }
  elements.graphMeta.textContent = `Showing ${nodeIds.length} of ${totalClasses} classes, ${edges.length} edges`;

  const remaining = new Set(nodeIds);
  const totalScores = {};
  nodeIds.forEach((i) => {
    let sum = 0;
    nodeIds.forEach((j) => {
      if (i !== j) {
        sum += scoreBetween(i, j);
      }
    });
    totalScores[i] = sum;
  });

  const order = [];
  const start = nodeIds.reduce((best, i) =>
    totalScores[i] > totalScores[best] ? i : best
  , nodeIds[0]);
  order.push(start);
  remaining.delete(start);

  while (remaining.size > 0) {
    const last = order[order.length - 1];
    let next = null;
    let bestScore = -1;
    remaining.forEach((candidate) => {
      const score = scoreBetween(last, candidate);
      if (score > bestScore) {
        bestScore = score;
        next = candidate;
      }
    });
    if (next === null) {
      next = remaining.values().next().value;
    }
    order.push(next);
    remaining.delete(next);
  }

  const width = container.clientWidth || 900;
  const height = Math.max(880, Math.round(width * 0.9));
  const padding = 100;
  const centerX = width / 2;
  const centerY = height / 2;
  const baseRadius = Math.min(width, height) / 2 - padding;
  const radius = Math.max(190, baseRadius * 0.82);

  const nodes = order.map((id, idx) => {
    const angle = (idx / order.length) * Math.PI * 2 - Math.PI / 2;
    return {
      id,
      short: state.classData[id].short,
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle),
    };
  });

  const nodeById = new Map(nodes.map((n) => [n.id, n]));

  const svg = d3
    .select(container)
    .append("svg")
    .attr("width", width)
    .attr("height", height);

  const rateExtent = d3.extent(edges, (d) => d.avgRate);
  const minRate = Number.isFinite(rateExtent[0]) ? rateExtent[0] : 0;
  const maxRate = Number.isFinite(rateExtent[1]) ? rateExtent[1] : 1;
  const domain = minRate === maxRate ? [0, maxRate || 1] : [minRate, maxRate];
  const color = d3
    .scaleLinear()
    .domain(domain)
    .range(["#bfe4ff", "#0b3a8e"])
    .interpolate(d3.interpolateRgb);
  const widthScale = d3.scaleLinear().domain(domain).range([6, 14]);

  const legendWidth = 180;
  const legendHeight = 10;
  const legendX = 24;
  const legendY = 20;
  const legendId = "edge-gradient";
  const defs = svg.append("defs");
  const gradient = defs
    .append("linearGradient")
    .attr("id", legendId)
    .attr("x1", "0%")
    .attr("x2", "100%")
    .attr("y1", "0%")
    .attr("y2", "0%");
  gradient.append("stop").attr("offset", "0%").attr("stop-color", color(domain[0]));
  gradient.append("stop").attr("offset", "100%").attr("stop-color", color(domain[1]));
  svg
    .append("rect")
    .attr("x", legendX)
    .attr("y", legendY)
    .attr("width", legendWidth)
    .attr("height", legendHeight)
    .attr("fill", `url(#${legendId})`)
    .attr("stroke", "#1f2a44")
    .attr("stroke-width", 1);
  svg
    .append("text")
    .attr("x", legendX)
    .attr("y", legendY - 6)
    .attr("font-size", 11)
    .attr("fill", "#1f2a44")
    .text("Avg confusion");
  svg
    .append("text")
    .attr("x", legendX)
    .attr("y", legendY + legendHeight + 14)
    .attr("font-size", 10)
    .attr("fill", "#1f2a44")
    .text(`${(domain[0] * 100).toFixed(1)}%`);
  svg
    .append("text")
    .attr("x", legendX + legendWidth)
    .attr("y", legendY + legendHeight + 14)
    .attr("font-size", 10)
    .attr("fill", "#1f2a44")
    .attr("text-anchor", "end")
    .text(`${(domain[1] * 100).toFixed(1)}%`);

  const link = svg
    .append("g")
    .attr("stroke-linecap", "round")
    .selectAll("line")
    .data(edges)
    .enter()
    .append("line")
    .attr("stroke", (d) => color(d.avgRate))
    .attr("stroke-width", (d) => widthScale(d.avgRate))
    .attr("stroke-opacity", 0.85)
    .attr("x1", (d) => {
      const src = nodeById.get(d.source);
      const tgt = nodeById.get(d.target);
      const dx = tgt.x - src.x;
      const dy = tgt.y - src.y;
      const len = Math.hypot(dx, dy) || 1;
      const offset = 8 * (d.source < d.target ? 1 : -1);
      const ox = (-dy / len) * offset;
      return src.x + ox;
    })
    .attr("y1", (d) => {
      const src = nodeById.get(d.source);
      const tgt = nodeById.get(d.target);
      const dx = tgt.x - src.x;
      const dy = tgt.y - src.y;
      const len = Math.hypot(dx, dy) || 1;
      const offset = 8 * (d.source < d.target ? 1 : -1);
      const oy = (dx / len) * offset;
      return src.y + oy;
    })
    .attr("x2", (d) => {
      const src = nodeById.get(d.source);
      const tgt = nodeById.get(d.target);
      const dx = tgt.x - src.x;
      const dy = tgt.y - src.y;
      const len = Math.hypot(dx, dy) || 1;
      const offset = 8 * (d.source < d.target ? 1 : -1);
      const ox = (-dy / len) * offset;
      return tgt.x + ox;
    })
    .attr("y2", (d) => {
      const src = nodeById.get(d.source);
      const tgt = nodeById.get(d.target);
      const dx = tgt.x - src.x;
      const dy = tgt.y - src.y;
      const len = Math.hypot(dx, dy) || 1;
      const offset = 8 * (d.source < d.target ? 1 : -1);
      const oy = (dx / len) * offset;
      return tgt.y + oy;
    })
    .on("mousemove", (event, d) => {
      const avgPct = (d.avgRate * 100).toFixed(2);
      const src = state.classData[d.source].short;
      const tgt = state.classData[d.target].short;
      const dirPct = totals[d.source] > 0 ? (d.rateIJ * 100).toFixed(2) : "0.00";
      const revPct = totals[d.target] > 0 ? (d.rateJI * 100).toFixed(2) : "0.00";
      showTooltip(
        `Avg confusion: ${avgPct}%<br>${src} -> ${tgt}: ${d.countIJ} (${dirPct}%)<br>` +
          `${tgt} -> ${src}: ${d.countJI} (${revPct}%)`,
        event
      );
    })
    .on("mouseleave", hideTooltip);

  const nodeGroup = svg.append("g").selectAll("g").data(nodes).enter().append("g");
  const nodeRadius = 20;
  nodeGroup
    .append("circle")
    .attr("r", nodeRadius)
    .attr("fill", colors.val)
    .attr("stroke", "#ffffff")
    .attr("stroke-width", 1.2)
    .attr("cx", (d) => d.x)
    .attr("cy", (d) => d.y);

  nodeGroup
    .append("text")
    .attr("x", (d) => d.x)
    .attr("y", (d) => d.y + 4)
    .attr("text-anchor", "middle")
    .attr("font-size", 11)
    .attr("fill", "#000000")
    .text((d) => d.short);
}

function initTabs() {
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", () => {
      const tab = button.dataset.tab;
      state.activeTab = tab;
      document.querySelectorAll(".tab-button").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.tab === tab);
      });
      document.querySelectorAll(".tab-panel").forEach((panel) => {
        panel.classList.toggle("active", panel.id === `tab-${tab}`);
      });
      elements.filterRow.classList.toggle("hidden", tab === "history" || tab === "graph");
      elements.graphFilterRow.classList.toggle("hidden", tab !== "graph");
      renderActiveTab();
    });
  });
}

function initRunMenu() {
  elements.runButton.addEventListener("click", () => {
    elements.runMenu.classList.toggle("hidden");
  });

  window.addEventListener("click", (event) => {
    if (
      !elements.runButton.contains(event.target) &&
      !elements.runMenu.contains(event.target)
    ) {
      elements.runMenu.classList.add("hidden");
    }
  });
}

function initFilter() {
  elements.filterInput.addEventListener("input", () => {
    state.filterMaxAcc = Number(elements.filterInput.value);
    elements.filterValue.textContent = String(state.filterMaxAcc);
    if (state.activeTab === "per-class" || state.activeTab === "matrix") {
      renderActiveTab();
    }
  });
}

function initGraphFilter() {
  elements.graphFilterInput.addEventListener("input", () => {
    state.graphMinConfusions = Number(elements.graphFilterInput.value);
    elements.graphFilterValue.textContent = String(state.graphMinConfusions);
    if (state.activeTab === "graph") {
      renderActiveTab();
    }
  });
  if (elements.graphEdgeToggle) {
    state.graphShowAllEdges = elements.graphEdgeToggle.checked;
    elements.graphEdgeToggle.addEventListener("change", () => {
      state.graphShowAllEdges = elements.graphEdgeToggle.checked;
      if (state.activeTab === "graph") {
        renderActiveTab();
      }
    });
  }
}

function initSelectionControls() {
  if (elements.selectAll) {
    elements.selectAll.addEventListener("click", () => {
      setSelection(state.classData.map((d) => d.index));
    });
  }
  if (elements.clearSelection) {
    elements.clearSelection.addEventListener("click", () => {
      setSelection([]);
    });
  }
  if (elements.selectVisible) {
    elements.selectVisible.addEventListener("click", () => {
      setSelection(filteredClasses().map((d) => d.index));
    });
  }
}

window.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initRunMenu();
  initFilter();
  initGraphFilter();
  initSelectionControls();
  loadRuns();
});
