"use strict";

const el = (id) => document.getElementById(id);
const parameter = (key) => el(`param-${key}`);
const form = el("settings");
const uniformForm = el("uniform-settings");
const controls = el("controls");
const canvas = el("heatmap");
const codeInputs = ["distance", "rounds", "shots", "seed"];
const resultActions = ["save-png", "save-csv", "save-stim", "open-circuit"];
const seismic = [[0, 0, 77], [0, 0, 255], [255, 255, 255], [255, 0, 0], [128, 0, 0]];
const timingFields = [
  ["r_gate", "Single-qubit rotation (µs)"], ["ms_gate", "MS gate (µs)"],
  ["reset", "Reset (µs)"], ["measure", "Measure (µs)"],
  ["split", "Split (µs)"], ["merge", "Merge (µs)"],
  ["swap", "Swap (µs)"], ["cool", "Cooling (µs)"],
  ["shuttle_speed", "Shuttle speed (µm/µs)"], ["shuttle_base_distance", "Base distance (µm)"],
  ["shuttle_ion_factor", "Distance reduction (µm/ion)"],
];
const thermalFields = [
  ["Shuttle", "phonon_shuttle_axial_per_180um", "phonon_shuttle_transversal", "phonon_shuttle_transversal2"],
  ["Split", "phonon_split_axial", "phonon_split_transversal", "phonon_split_transversal2"],
  ["Merge", "phonon_merge_axial", "phonon_merge_transversal", "phonon_merge_transversal2"],
  ["Swap", "phonon_swap_axial", "phonon_swap_transversal", "phonon_swap_transversal2"],
  ["After cooling", "phonon_cool_residual_axial", "phonon_cool_residual_transversal", "phonon_cool_residual_transversal2"],
  ["Background heating", "heating_rate_axial_per_s", "heating_rate_transversal_per_s", "heating_rate_transversal2_per_s"],
];
const noiseFields = [
  ["p_R", "Rotation error"], ["p_meas", "Measurement error"],
  ["p_reset", "Reset error"], ["gamma_phi_per_s", "Idle dephasing (1/s)"],
];

let defaults;
let lastResult;
let lastPreset;
let plot;
let slots;

function addField(parent, key, title, { max, hint, optional = false } = {}) {
  const label = document.createElement("label");
  label.textContent = title;
  const input = document.createElement("input");
  input.id = `param-${key}`;
  input.type = "number";
  input.min = "0";
  input.step = "any";
  input.required = !optional;
  if (max !== undefined) input.max = max;
  if (hint) input.placeholder = hint;
  label.append(input);
  parent.append(label);
}

function buildFields() {
  for (const [key, title] of timingFields) addField(el("timing-fields"), key, title);
  for (const [title, axial, transverse, second] of thermalFields) {
    const row = document.createElement("section");
    row.className = "thermal-row";
    const heading = document.createElement("h3");
    heading.textContent = title;
    const fields = document.createElement("div");
    fields.className = "fields";
    addField(fields, axial, title === "Shuttle" ? "Axial / 180 µm" : "Axial");
    addField(fields, transverse, "Transverse 1");
    addField(fields, second, "Transverse 2", { optional: true, hint: "As T1" });
    row.append(heading, fields);
    el("thermal-fields").append(row);
  }
  for (const [key, title] of noiseFields) {
    addField(el("noise-fields"), key, title, { max: key.startsWith("p_") ? 1 : undefined });
  }
}

function readZones() {
  return el("zones").value.trim().toUpperCase().split(/[\s,]+/).map((zone) => {
    const match = /^([TML]):([0-9]+)$/.exec(zone);
    if (!match) throw new Error("Use type:capacity for each zone, such as T:20 M:20 T:20 L:4.");
    return [match[1], Number(match[2])];
  });
}

function readIds(value) {
  const clean = value.trim();
  if (!clean) return [];
  const ids = clean.split(/[\s,]+/);
  if (ids.some((id) => !/^\d+$/.test(id) || !Number.isSafeInteger(Number(id)))) {
    throw new Error("Use nonnegative integer ion IDs separated by spaces or commas.");
  }
  return ids.map(Number);
}

function readChains() {
  const value = el("chains").value.trim();
  if (!value) return null;
  const chains = {};
  for (const line of value.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const match = /^\s*z?(\d+)\s*:(.*)$/i.exec(line);
    if (!match) throw new Error("Enter starting chains as zone: ion ion …, one zone per line.");
    const zone = Number(match[1]);
    if (zone in chains) throw new Error(`Enter zone ${zone} only once.`);
    chains[zone] = readIds(match[2]);
  }
  return chains;
}

function setLayout(lizCount) {
  const d = Number(el("distance").value);
  const capacity = Math.max(defaults.zones[1][1], Math.ceil(d * d / 2) + 5);
  const zones = defaults.zones.concat(lizCount === 2 ? defaults.zones.slice(3) : []);
  el("zones").value = zones.map(([kind, size]) => `${kind}:${kind === "L" ? size : Math.max(size, capacity)}`).join(" ");
  lastPreset = { lizCount, value: el("zones").value };
  updateTrack();
}

function updateTrack() {
  const track = el("track");
  track.replaceChildren();
  let zones = [];
  try {
    zones = readZones();
  } catch {
    // Keep the layout editable while a zone is only partially typed.
  }
  const count = zones.filter(([type]) => type === "L").length;
  zones.forEach(([type, capacity], index) => {
    const zone = document.createElement("span");
    zone.className = "zone";
    zone.dataset.kind = type;
    zone.title = `z${index}, ${capacity} ions`;
    const kind = document.createElement("b");
    kind.textContent = type;
    const id = document.createElement("small");
    id.textContent = `z${index}`;
    zone.append(kind, id);
    track.append(zone);
  });
  document.querySelectorAll("[data-liz]").forEach((button) => {
    button.setAttribute("aria-pressed", String(Number(button.dataset.liz) === count));
  });
  const dataCount = Number(el("distance").value) ** 2;
  el("rounds").max = Math.floor(defaults.max_detectors / (dataCount - 1));
  el("ion-ids").textContent = `Data ions 0–${dataCount - 1} · Ancilla ion${count === 2 ? "s " + dataCount + "–" + (dataCount + 1) : " " + dataCount}`;
  el("placement-mode").textContent = el("chains").value.trim() ? "Custom" : "Automatic";
}

function setStatus(message, isError = false) {
  el("status").textContent = message;
  el("status").classList.toggle("error", isError);
}

function resetDefaults() {
  form.reset();
  uniformForm.reset();
  for (const fields of [defaults.timing, defaults.noise]) {
    for (const [key, value] of Object.entries(fields)) parameter(key).value = value ?? "";
  }
  el("distance").value = defaults.distances[0];
  setLayout(1);
  setStatus(lastResult ? "Defaults restored. Run again to update the result." : "");
}

function codeSettings() {
  return Object.fromEntries(codeInputs.map((id) => [id, Number(el(id).value)]));
}

function settingsPayload() {
  const numericValue = (input) => input.value === "" ? null : Number(input.value);
  const value = (id) => numericValue(el(id));
  const fieldValues = (keys) => Object.fromEntries(
    keys.map((key) => [key, numericValue(parameter(key))]),
  );
  return {
    ...codeSettings(),
    zones: readZones(),
    chains: readChains(),
    timing: fieldValues(Object.keys(defaults.timing)),
    noise: fieldValues(Object.keys(defaults.noise)),
    lookup_mode: el("lookup-mode").value,
    cooling: {
      threshold: value("threshold"),
      after_measure_us: value("after-measure"),
      resident_ids: readIds(el("coolants").value),
      recool_us: value("recool"),
    },
  };
}

function palette(value) {
  if (value === null) return seismic[2];
  const position = 2 * (Math.max(-1, Math.min(1, value)) + 1);
  const index = Math.min(3, Math.floor(position));
  const weight = position - index;
  return seismic[index].map((channel, i) => Math.round(channel + weight * (seismic[index + 1][i] - channel)));
}

function runLabel(result) {
  const rounds = result.rounds === 1 ? "round" : "rounds";
  return `Distance ${result.distance} · ${result.rounds} ${rounds}`;
}

function drawHeatmap(result, target = canvas, width = Math.min(768, Math.floor(canvas.clientWidth))) {
  const compact = width < 550;
  const uniform = result.uniform_p != null;
  const x = compact ? 52 : 77;
  const y = (compact ? 42 : 57) + (uniform ? 17 : 0);
  const size = Math.max(100, width - x - (compact ? 16 : 162));
  const height = y + size + (compact ? 126 : 93);
  if (target === canvas) plot = { x, y, size, width, height };
  const n = slots.length;
  const ctx = target.getContext("2d");
  target.width = width * 2;
  target.height = height * 2;
  ctx.scale(2, 2);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, width, height);

  let limit = 0;
  for (const i of slots) {
    for (const j of slots) limit = Math.max(limit, Math.abs(pairValue(i, j)));
  }
  limit ||= .01;
  const raster = document.createElement("canvas");
  raster.width = raster.height = n;
  const rasterCtx = raster.getContext("2d");
  const pixels = rasterCtx.createImageData(n, n);
  slots.forEach((left, i) => slots.forEach((right, j) => {
    const value = pairValue(left, right);
    const rgb = palette(value === null ? null : value / limit);
    pixels.data.set([...rgb, 255], ((n - 1 - i) * n + j) * 4);
  }));
  rasterCtx.putImageData(pixels, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(raster, x, y, size, size);

  const block = size / result.checks.length;
  ctx.beginPath();
  for (let index = 1; index < result.checks.length; index++) {
    const offset = index * block;
    ctx.moveTo(x + offset, y);
    ctx.lineTo(x + offset, y + size);
    ctx.moveTo(x, y + size - offset);
    ctx.lineTo(x + size, y + size - offset);
  }
  ctx.strokeStyle = "#808080";
  ctx.lineWidth = Math.min(1.5, block / 4);
  ctx.stroke();
  ctx.strokeStyle = "#111";
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y, size, size);
  ctx.fillStyle = "#111";
  ctx.font = compact ? "14px system-ui, sans-serif" : "18px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Two-point correlation matrix pᵢⱼ", x + size / 2, y - (uniform ? 32 : 15), size);
  if (uniform) {
    ctx.font = "13px system-ui, sans-serif";
    ctx.fillText(`Uniform DEM, p = ${result.uniform_p}`, x + size / 2, y - 13, size);
  }
  ctx.font = compact ? "12px system-ui, sans-serif" : "14px system-ui, sans-serif";
  const step = Math.max(1, Math.ceil((compact ? 24 : 25) / block));
  for (let index = 0; index < result.checks.length; index += step) {
    const label = result.checks[index].label;
    const offset = (index + .5) * block;
    ctx.beginPath();
    ctx.moveTo(x + offset, y + size);
    ctx.lineTo(x + offset, y + size + 5);
    ctx.moveTo(x, y + size - offset);
    ctx.lineTo(x - 5, y + size - offset);
    ctx.stroke();
    ctx.save();
    ctx.translate(x + offset, y + size + 11);
    ctx.rotate(-Math.PI / 4);
    ctx.textAlign = "right";
    ctx.fillText(label, 0, 3);
    ctx.restore();
    ctx.textAlign = "right";
    ctx.fillText(label, x - 9, y + size - offset + 4);
  }
  ctx.font = compact ? "13px system-ui, sans-serif" : "16px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Stabilizer check", x + size / 2, y + size + (compact ? 50 : 59));
  ctx.save();
  ctx.translate(compact ? 13 : 22, y + size / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Stabilizer check", 0, 0);
  ctx.restore();

  const decimals = Math.max(2, Math.ceil(-Math.log10(limit)) + 1);
  const tick = (fraction) => (limit * fraction).toFixed(Math.min(6, decimals)).replace("-", "−");
  if (compact) {
    const legendY = y + size + 81;
    const legend = colorGradient(ctx.createLinearGradient(x, 0, x + size, 0), -1, 1);
    ctx.fillStyle = legend;
    ctx.fillRect(x, legendY, size, 12);
    ctx.strokeRect(x, legendY, size, 12);
    ctx.fillStyle = "#111";
    ctx.textAlign = "center";
    ctx.fillText("pᵢⱼ", x + size / 2, legendY - 6);
    ctx.textAlign = "left";
    ctx.fillText(tick(-1), x, legendY + 29);
    ctx.textAlign = "center";
    ctx.fillText(tick(0), x + size / 2, legendY + 29);
    ctx.textAlign = "right";
    ctx.fillText(tick(1), x + size, legendY + 29);
    return;
  }

  const legendX = x + size + 24;
  const legend = colorGradient(ctx.createLinearGradient(0, y, 0, y + size), 1, -1);
  ctx.fillStyle = legend;
  ctx.fillRect(legendX, y, 18, size);
  ctx.strokeRect(legendX, y, 18, size);
  ctx.fillStyle = "#111";
  ctx.textAlign = "left";
  ctx.font = "14px system-ui, sans-serif";
  for (const [fraction, level] of [[0, 1], [.25, .5], [.5, 0], [.75, -.5], [1, -1]]) {
    const tickY = y + size * fraction;
    ctx.beginPath();
    ctx.moveTo(legendX + 18, tickY);
    ctx.lineTo(legendX + 23, tickY);
    ctx.stroke();
    ctx.fillText(tick(level), legendX + 29, tickY + 5);
  }
  ctx.save();
  ctx.translate(legendX + 101, y + size / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("pᵢⱼ", 0, 0);
  ctx.restore();
}

function colorGradient(gradient, start, end) {
  for (let i = 0; i <= 4; i++) {
    gradient.addColorStop(i / 4, `rgb(${palette(start + (end - start) * i / 4).join(",")})`);
  }
  return gradient;
}

function pairValue(i, j) {
  const left = i.measurement?.detector;
  const right = j.measurement?.detector;
  return left == null || right == null ? null : lastResult.matrix[left][right];
}

function showResult(result) {
  const uniform = result.uniform_p != null;
  lastResult = result;
  slots = result.checks.flatMap(check => check.rounds.map((measurement, round) => ({
    label: check.label, round: round + 1, measurement,
  })));
  clearPair();
  const noiseLabel = uniform ? `uniform DEM (p = ${result.uniform_p})` : "per-ion idle noise";
  el("result-caption").textContent = `${runLabel(result)} · ${noiseLabel}`;
  el("detectors").textContent = result.detectors.toLocaleString();
  el("sampled").textContent = result.shots.toLocaleString();
  el("round-duration").hidden = uniform;
  el("round-time").textContent = uniform ? "" : `${(result.duration_ms / result.rounds).toFixed(2)} ms`;
  el("empty").hidden = true;
  el("metrics").hidden = false;
  el("plot-area").hidden = false;
  drawHeatmap(result);
  resultActions.forEach((id) => { el(id).disabled = false; });
  el("save-dem").hidden = !uniform;
  prepareInspector(result);
  const warning = el("lookup-warning");
  warning.hidden = !result.clipped_gates;
  warning.textContent = result.clipped_gates
    ? `The MS lookup capped ${result.clipped_gates.toLocaleString()} gates at ${result.lookup_ceiling} phonons. Peak at an MS gate was ${result.max_phonon.toFixed(2)}.`
    : "";
}

function fillSelect(select, labels) {
  select.replaceChildren(...labels.map((label, index) => new Option(label, index)));
}

function prepareInspector(result) {
  fillSelect(el("circuit-round"), Array.from({ length: result.rounds }, (_, index) => `Round ${index + 1}`));
  fillSelect(el("circuit-check"), result.checks.map(check => check.label));
  const select = el("circuit-view");
  for (const [view, available] of [["schedule", result.schedules != null], ["dem", result.dem != null]]) {
    const option = select.querySelector(`[value="${view}"]`);
    option.disabled = option.hidden = !available;
    if (!available && select.value === view) select.value = "diagram";
  }
  el("circuit-inspector").hidden = false;
  renderInspector();
}

function renderGates(check, measurement) {
  const qubits = [...check.data, measurement.ancilla].sort((a, b) => a - b);
  const diagram = document.createElement("div");
  diagram.className = "gate-diagram";
  diagram.style.setProperty("--wires", qubits.length);
  const labels = document.createElement("div");
  labels.className = "wire-labels";
  diagram.append(labels);
  for (const ion of qubits) {
    const label = document.createElement("span");
    label.textContent = `q${ion}${ion === measurement.ancilla ? " (anc.)" : ""}`;
    labels.append(label);
  }
  for (const [gate, targets] of measurement.gates) {
    const column = document.createElement("div");
    column.className = "gate-step";
    column.title = `${gate} ${targets.join(" ")}`;
    if (gate === "CX") {
      const wires = targets.map(ion => qubits.indexOf(ion));
      column.style.setProperty("--link-start", `${(Math.min(...wires) + .5) * 2.2}rem`);
      column.style.setProperty("--link-size", `${Math.abs(wires[0] - wires[1]) * 2.2}rem`);
    }
    for (const ion of qubits) {
      const wire = document.createElement("div");
      wire.className = "gate-wire";
      const target = targets.indexOf(ion);
      if (target !== -1) {
        const symbol = document.createElement("span");
        symbol.className = "gate-symbol";
        if (gate === "CX") {
          symbol.classList.add(target === 0 ? "gate-control" : "gate-target");
          symbol.textContent = target === 0 ? "●" : "⊕";
        } else {
          symbol.classList.add("gate-box");
          symbol.textContent = gate;
        }
        wire.append(symbol);
      }
      column.append(wire);
    }
    diagram.append(column);
  }
  el("circuit-diagram").replaceChildren(diagram);
}

function renderInspector() {
  const view = el("circuit-view").value;
  const round = Number(el("circuit-round").value);
  const isDiagram = view === "diagram";
  el("circuit-round-control").hidden = !isDiagram && view !== "schedule";
  el("circuit-check-control").hidden = !isDiagram;
  el("circuit-diagram").hidden = !isDiagram;
  el("circuit-source").hidden = isDiagram;
  if (isDiagram) {
    const check = lastResult.checks[Number(el("circuit-check").value)];
    const measurement = check.rounds[round];
    if (measurement) {
      const detector = measurement.detector === null ? "No detector for this measurement" : `Stim detector D${measurement.detector}`;
      el("circuit-info").textContent = `Ancilla q${measurement.ancilla} · Data ions ${check.data.join(", ")} · ${detector}`;
      renderGates(check, measurement);
    } else {
      el("circuit-info").textContent = "This check was not measured in the selected round.";
      el("circuit-diagram").replaceChildren();
    }
  } else {
    const isSchedule = view === "schedule";
    const description = {
      schedule: "Native gates and ion transport for the selected round.",
      logical_stim: "Logical gates and detector definitions before noise injection.",
      stim: lastResult.dem != null ? "Circuit used to derive the uniform detector error model." : "Exact circuit sampled for this run.",
      dem: "Exact detector error model sampled for this run.",
    };
    el("circuit-info").textContent = description[view];
    const source = el("circuit-source");
    source.textContent = isSchedule ? lastResult.schedules[round] : lastResult[view];
    source.scrollTop = source.scrollLeft = 0;
  }
}

el("open-circuit").addEventListener("click", () => {
  const inspector = el("circuit-inspector");
  inspector.open = true;
  inspector.scrollIntoView({ block: "start" });
  el("circuit-view").focus({ preventScroll: true });
});
for (const id of ["circuit-view", "circuit-round", "circuit-check"]) {
  el(id).addEventListener("change", renderInspector);
}

async function run(event) {
  event.preventDefault();
  const uniform = event.currentTarget === uniformForm;
  if (uniform && !codeInputs.every((id) => el(id).reportValidity())) return;
  const button = el(uniform ? "run-uniform" : "run");
  const buttonLabel = button.textContent;
  controls.disabled = true;
  button.textContent = "Running…";
  const status = uniform ? "Building and sampling the uniform detector error model…" : "Running the hardware model…";
  el("empty-message").textContent = status;
  setStatus(status);
  try {
    const payload = uniform ? { ...codeSettings(), p: el("uniform-p").valueAsNumber } : settingsPayload();
    if (!uniform && payload.distance >= 11) setStatus("Running the hardware model… Larger circuits can take several minutes.");
    const calibration = uniform ? null : el("lookup-file").files[0];
    if (calibration) {
      if (calibration.size > 350_000) throw new Error("Use a calibration CSV smaller than 350 KB.");
      payload.lookup_csv = await calibration.text();
    }
    const response = await fetch(uniform ? "/api/uniform-dem" : "/api/simulate", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error);
    showResult(result);
    setStatus("Simulation complete.");
  } catch (error) {
    el("empty-message").textContent = "Run a simulation to generate the detector matrix.";
    setStatus(error.message + (lastResult ? "\nThe plot shows the previous run." : ""), true);
  } finally {
    controls.disabled = false;
    button.textContent = buttonLabel;
  }
}

function download(blob, extension, result = lastResult) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  const suffix = result.uniform_p == null ? "" : `-uniform-p${result.uniform_p}`;
  link.download = `theata-d${result.distance}-r${result.rounds}-pij${suffix}.${extension}`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

el("save-png").addEventListener("click", () => {
  const result = lastResult;
  const figure = document.createElement("canvas");
  drawHeatmap(result, figure, 768);
  figure.toBlob((blob) => { if (blob) download(blob, "png", result); }, "image/png");
});
el("save-csv").addEventListener("click", () => {
  const headers = lastResult.matrix.map((_, index) => index);
  const rows = lastResult.matrix.map((row, index) => [index, ...row].join(","));
  download(new Blob([["detector", ...headers].join(","), "\n", rows.join("\n"), "\n"],
    { type: "text/csv" }), "csv");
});
el("save-stim").addEventListener("click", () => {
  download(new Blob([lastResult.stim, "\n"], { type: "text/plain" }), "stim");
});
el("save-dem").addEventListener("click", () => {
  download(new Blob([lastResult.dem, "\n"], { type: "text/plain" }), "dem");
});

function clearPair() {
  el("pair-info").textContent = "Point at or tap a cell to inspect its detector pair.";
}
canvas.addEventListener("pointerleave", (event) => {
  if (event.pointerType !== "touch") clearPair();
});
window.addEventListener("resize", () => {
  if (lastResult) drawHeatmap(lastResult);
});
function showPair(event) {
  if (!lastResult) return;
  const rect = canvas.getBoundingClientRect();
  const n = slots.length;
  const row = Math.floor(((event.clientY - rect.top) * plot.height / rect.height - plot.y) * n / plot.size);
  const i = n - 1 - row;
  const j = Math.floor(((event.clientX - rect.left) * plot.width / rect.width - plot.x) * n / plot.size);
  if (i < 0 || j < 0 || i >= n || j >= n) {
    clearPair();
    return;
  }
  const describe = (slot) => {
    const detector = slot.measurement?.detector;
    return `${slot.label} r${slot.round}${detector == null ? "" : ` (D${detector})`}`;
  };
  const value = pairValue(slots[i], slots[j]);
  el("pair-info").textContent = `${describe(slots[i])} × ${describe(slots[j])} · ${value === null ? "Undefined" : Number(value.toPrecision(4))}`;
}
canvas.addEventListener("pointermove", showPair);
canvas.addEventListener("pointerdown", showPair);

form.addEventListener("submit", run);
uniformForm.addEventListener("submit", run);
form.addEventListener("invalid", (event) => {
  const details = event.target.closest("details");
  if (details) details.open = true;
}, true);
form.addEventListener("input", () => {
  if (lastResult) setStatus("Settings changed. Run again to update the result.");
});
el("zones").addEventListener("input", updateTrack);
el("chains").addEventListener("input", updateTrack);
el("distance").addEventListener("change", () => {
  if (lastPreset.value === el("zones").value) setLayout(lastPreset.lizCount);
  else updateTrack();
});
el("reset-defaults").addEventListener("click", resetDefaults);
document.querySelectorAll("[data-liz]").forEach((button) => {
  button.addEventListener("click", () => {
    setLayout(Number(button.dataset.liz));
    if (lastResult) setStatus("Layout changed. Run again to update the result.");
  });
});

async function init() {
  try {
    const response = await fetch("/api/defaults");
    if (!response.ok) throw new Error("Could not load hardware defaults.");
    defaults = await response.json();
    buildFields();
    for (const distance of defaults.distances) {
      const option = document.createElement("option");
      option.value = distance;
      option.textContent = `d = ${distance}`;
      el("distance").append(option);
    }
    resetDefaults();
    controls.disabled = false;
  } catch (error) {
    setStatus(`${error.message} Start the THeatA server and reload the page.`, true);
  }
}

init();
