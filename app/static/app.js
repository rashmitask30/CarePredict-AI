// Plain JavaScript frontend: builds the form from /api/options and calls /api/predict.
const NUM = [
  ["time_in_hospital", "Days in hospital", 1, 14],
  ["num_lab_procedures", "Lab procedures", 0, 150],
  ["num_procedures", "Procedures (non-lab)", 0, 10],
  ["num_medications", "Number of medications", 0, 100],
  ["number_outpatient", "Outpatient visits (prior year)", 0, 100],
  ["number_emergency", "Emergency visits (prior year)", 0, 100],
  ["number_inpatient", "Inpatient visits (prior year)", 0, 50],
  ["number_diagnoses", "Number of diagnoses", 1, 20],
];
const CAT = [
  ["age", "Age bracket"],
  ["admission_type_id", "Admission type"],
  ["admission_source_id", "Admission source"],
  ["discharge_disposition_id", "Discharge disposition"],
  ["diag_1_category", "Primary diagnosis group"],
  ["diag_2_category", "Secondary diagnosis group"],
  ["diag_3_category", "Additional diagnosis group"],
  ["A1Cresult", "HbA1c result"],
  ["max_glu_serum", "Max glucose serum"],
  ["insulin", "Insulin"],
  ["change", "Diabetes medication changed?"],
  ["diabetesMed", "Diabetes medication prescribed?"],
];
const EX = {
  low:  {age:"[40-50)", time_in_hospital:2, num_lab_procedures:30, num_procedures:0, num_medications:8, number_outpatient:0, number_emergency:0, number_inpatient:0, number_diagnoses:5, admission_type_id:3, discharge_disposition_id:1, admission_source_id:1, diag_1_category:"Circulatory", diag_2_category:"Endocrine_Metabolic", diag_3_category:"Unknown", A1Cresult:"Not measured", max_glu_serum:"Not measured", insulin:"No", change:"No", diabetesMed:"No"},
  high: {age:"[70-80)", time_in_hospital:9, num_lab_procedures:30, num_procedures:0, num_medications:25, number_outpatient:2, number_emergency:2, number_inpatient:3, number_diagnoses:9, admission_type_id:1, discharge_disposition_id:3, admission_source_id:7, diag_1_category:"Circulatory", diag_2_category:"Endocrine_Metabolic", diag_3_category:"Unknown", A1Cresult:">8", max_glu_serum:"Not measured", insulin:"Up", change:"Ch", diabetesMed:"Yes"},
};
const NUMERIC_IDS = ["admission_type_id", "admission_source_id", "discharge_disposition_id"];
let OPTIONS = null;

function el(id) { return document.getElementById(id); }

async function init() {
  const r = await fetch("/api/options");
  const data = await r.json();
  OPTIONS = data.options;
  const form = el("form");
  NUM.forEach(([id, label, min, max]) => {
    form.insertAdjacentHTML("beforeend",
      `<div><label for="${id}">${label}</label><input type="number" id="${id}" min="${min}" max="${max}" value="${Math.round(data.defaults[id])}"></div>`);
  });
  CAT.forEach(([id, label]) => {
    const src = id.startsWith("diag_") ? OPTIONS.diag_1_category : OPTIONS[id];
    const opts = src.map(o => typeof o === "object" ? `<option value="${o.value}">${o.label}</option>` : `<option value="${o}">${o}</option>`).join("");
    form.insertAdjacentHTML("beforeend", `<div><label for="${id}">${label}</label><select id="${id}">${opts}</select></div>`);
  });
  el("ex-low").onclick = () => fill(EX.low);
  el("ex-high").onclick = () => fill(EX.high);
  el("go").onclick = predict;
  fill(EX.low);
}

function fill(values) {
  Object.entries(values).forEach(([k, v]) => { el(k).value = v; });
}

function collect() {
  const body = {};
  NUM.forEach(([id]) => body[id] = parseInt(el(id).value, 10));
  CAT.forEach(([id]) => body[id] = NUMERIC_IDS.includes(id) ? parseInt(el(id).value, 10) : el(id).value);
  return body;
}

async function predict() {
  el("error").textContent = "";
  const body = collect();
  const res = await fetch("/api/predict", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  if (!res.ok) {
    const err = await res.json();
    el("error").textContent = "Please check the inputs: " + JSON.stringify(err.detail).slice(0, 200);
    return;
  }
  const d = await res.json();
  el("result").hidden = false;
  const t = el("tier");
  t.textContent = d.risk_category + " risk";
  t.className = "tier " + d.risk_category;
  document.querySelector(".prob").innerHTML =
    `Model risk score: <b>${(d.risk_score * 100).toFixed(1)}</b> / 100 &nbsp;|&nbsp; ` +
    `Estimated 30-day readmission probability: <b>${(d.estimated_30d_readmission_probability * 100).toFixed(1)}%</b> ` +
    `(average for all diabetic encounters is about 11%)`;
  el("guidance").textContent = d.guidance;
  el("factors").innerHTML = d.top_factors.map(f => {
    const shown = body[f.feature] !== undefined ? body[f.feature] : f.value;
    return `<li><b>${f.feature}</b> = ${shown} ${f.direction === "raises" ? "increases" : "decreases"} the predicted risk</li>`;
  }).join("");
  el("calnote").textContent = "Factors come from the " + d.model + " model's own contribution values for this prediction. They show association in historical data, not cause and effect.";
  document.querySelector(".disc").textContent = d.disclaimer;
  el("result").scrollIntoView({behavior: "smooth"});
}

init();
