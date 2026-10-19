const healthStatus = document.querySelector("#health-status");
const modelChip = document.querySelector("#model-chip");
const heroModel = document.querySelector("#hero-model");
const heroDocuments = document.querySelector("#hero-documents");
const fileLimit = document.querySelector("#file-limit");
const fileInput = document.querySelector("#file-input");
const dropZone = document.querySelector("#drop-zone");
const uploadList = document.querySelector("#upload-list");
const documentCount = document.querySelector("#document-count");
const documentList = document.querySelector("#document-list");
const documentMessage = document.querySelector("#document-message");
const refreshDocuments = document.querySelector("#refresh-documents");
const askForm = document.querySelector("#ask-form");
const questionInput = document.querySelector("#question-input");
const askButton = document.querySelector("#ask-button");
const questionMessage = document.querySelector("#question-message");
const answerSection = document.querySelector("#answer-section");
const answerStatus = document.querySelector("#answer-status");
const answerText = document.querySelector("#answer-text");
const citationArea = document.querySelector("#citation-area");
const citationList = document.querySelector("#citation-list");
const citationCount = document.querySelector("#citation-count");
const flowTabs = [...document.querySelectorAll(".flow-tab")];
const flowState = document.querySelector("#flow-state");
const flowSummary = document.querySelector("#flow-summary");
const flowStages = document.querySelector("#flow-stages");
const flowFacts = document.querySelector("#flow-facts");
const traceScope = document.querySelector("#trace-scope");

const PIPELINES = {
  ingestion: {
    idleSummary: "Choose a document to awaken the ingestion pipeline.",
    ariaLabel: "Ingestion stages",
    stages: [
      ["Validation", "type · size · content"],
      ["Extraction", "PDF or UTF-8 text"],
      ["Chunking", "deterministic windows"],
      ["Embeddings", "local MiniLM vectors"],
      ["Qdrant indexing", "persistent local points"],
    ],
    facts: {
      Document: "Awaiting input",
      Chunks: "—",
      Identity: "—",
      Result: "—",
      Confirmation: "No request observed",
    },
  },
  qa: {
    idleSummary: "Ask a question to observe grounded answer construction.",
    ariaLabel: "Question answering stages",
    stages: [
      ["Retrieval", "ranked vector search"],
      ["Evidence selection", "bounded top-k passages"],
      ["Context assembly", "hard character budget"],
      ["Ollama generation", "installed local model"],
      ["Citation validation", "retrieved IDs only"],
      ["Answer", "grounded or abstained"],
    ],
    facts: {
      Model: "Awaiting system details",
      Retrieved: "—",
      Sources: "—",
      Scores: "—",
      Context: "—",
      Citations: "—",
    },
  },
};

const pipelineViews = {
  ingestion: {
    state: "idle",
    summary: PIPELINES.ingestion.idleSummary,
    stages: PIPELINES.ingestion.stages.map(() => "idle"),
    facts: { ...PIPELINES.ingestion.facts },
    scope: "No request observed",
  },
  qa: {
    state: "idle",
    summary: PIPELINES.qa.idleSummary,
    stages: PIPELINES.qa.stages.map(() => "idle"),
    facts: { ...PIPELINES.qa.facts },
    scope: "No request observed",
  },
};

let activePipeline = "ingestion";
let maxUploadBytes = 10 * 1024 * 1024;
let maxContextChars = 12000;
let ollamaModel = "Unknown local model";

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error(`The server returned an unreadable response (${response.status}).`);
  }
  if (!response.ok) {
    const detail = payload.detail;
    const message = typeof detail === "object" ? detail.message : detail;
    const error = new Error(message || payload.error || `Request failed (${response.status}).`);
    error.payload = payload;
    throw error;
  }
  return payload;
}

function renderFlow() {
  const definition = PIPELINES[activePipeline];
  const view = pipelineViews[activePipeline];
  flowState.className = `flow-state ${view.state}`;
  flowState.innerHTML = `<span></span>${escapeStatus(view.state)}`;
  flowSummary.textContent = view.summary;
  traceScope.textContent = view.scope;
  flowStages.setAttribute("aria-label", definition.ariaLabel);
  flowStages.replaceChildren();

  definition.stages.forEach(([name, description], index) => {
    const status = view.stages[index] || "idle";
    const stage = document.createElement("li");
    stage.className = `pipeline-stage ${status}`;
    stage.setAttribute("aria-label", `${name}: ${escapeStatus(status)}`);
    const top = document.createElement("div");
    top.className = "stage-top";
    const number = document.createElement("span");
    number.className = "stage-index";
    number.textContent = String(index + 1).padStart(2, "0");
    const icon = document.createElement("span");
    icon.className = "stage-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = stageIcon(status);
    const title = document.createElement("strong");
    title.textContent = name;
    const detail = document.createElement("small");
    detail.textContent = description;
    top.append(number, icon);
    stage.append(top, title, detail);
    flowStages.append(stage);
  });

  flowFacts.replaceChildren();
  Object.entries(view.facts).forEach(([label, value]) => {
    const wrapper = document.createElement("div");
    wrapper.className = "trace-fact";
    const term = document.createElement("dt");
    term.textContent = label;
    const description = document.createElement("dd");
    description.textContent = value;
    description.title = value;
    wrapper.append(term, description);
    flowFacts.append(wrapper);
  });
}

function stageIcon(status) {
  return {
    complete: "✓",
    active: "◇",
    pending: "…",
    failed: "!",
    skipped: "—",
    idle: "·",
  }[status] || "·";
}

function escapeStatus(status) {
  return status === "completed" ? "Completed" : status.charAt(0).toUpperCase() + status.slice(1);
}

function selectPipeline(name) {
  activePipeline = name;
  flowTabs.forEach((tab) => {
    const selected = tab.dataset.pipeline === name;
    tab.classList.toggle("active", selected);
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  renderFlow();
}

function setPipeline(name, update) {
  pipelineViews[name] = { ...pipelineViews[name], ...update };
  if (activePipeline === name) renderFlow();
}

flowTabs.forEach((tab) => {
  tab.addEventListener("click", () => selectPipeline(tab.dataset.pipeline));
  tab.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const next = tab.dataset.pipeline === "ingestion" ? "qa" : "ingestion";
    selectPipeline(next);
    document.querySelector(`[data-pipeline="${next}"]`).focus();
  });
});

async function checkHealth() {
  try {
    const health = await requestJson("/api/health");
    maxUploadBytes = health.max_upload_bytes;
    maxContextChars = health.max_context_chars;
    ollamaModel = health.ollama_model || ollamaModel;
    fileLimit.textContent = `PDF / TXT · up to ${formatBytes(maxUploadBytes)}`;
    modelChip.textContent = `Model · ${ollamaModel}`;
    heroModel.textContent = `${ollamaModel} · loopback only`;
    healthStatus.className = "system-status online";
    healthStatus.lastElementChild.textContent = "Local API ready";
    pipelineViews.qa.facts.Model = ollamaModel;
    renderFlow();
    await loadDocuments();
  } catch {
    healthStatus.className = "system-status offline";
    healthStatus.lastElementChild.textContent = "Local API unavailable";
    modelChip.textContent = "Model · unavailable";
    heroModel.textContent = "Local API unavailable";
  }
}

function formatBytes(bytes) {
  if (bytes >= 1024 * 1024) return `${Math.round(bytes / 1024 / 1024)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${bytes} B`;
}

function addUploadItem(file) {
  const item = document.createElement("div");
  item.className = "upload-item";
  const name = document.createElement("strong");
  name.textContent = file.name;
  const state = document.createElement("span");
  state.textContent = "Awaiting validation";
  item.append(name, state);
  uploadList.prepend(item);
  return { item, state };
}

function ingestionFacts(file, extras = {}) {
  return {
    Document: file.name,
    Chunks: extras.chunks || "—",
    Identity: extras.identity || "—",
    Result: extras.result || "Awaiting response",
    Confirmation: extras.confirmation || `${formatBytes(file.size)} submitted`,
  };
}

function failIngestion(file, error, row) {
  const code = error.payload?.detail?.code;
  let stages = ["failed", "skipped", "skipped", "skipped", "skipped"];
  if (code === "invalid_document") stages = ["complete", "failed", "skipped", "skipped", "skipped"];
  if (code === "index_error") stages = ["complete", "complete", "pending", "pending", "pending"];
  row.item.classList.add("error");
  row.state.textContent = error.message;
  setPipeline("ingestion", {
    state: "failed",
    summary: `The ingestion request failed: ${error.message}`,
    stages,
    facts: ingestionFacts(file, {
      result: code ? code.replaceAll("_", " ") : "Request failed",
      confirmation: "Downstream stages not confirmed",
    }),
    scope: code ? `API error · ${code}` : "No API confirmation",
  });
}

async function uploadFile(file) {
  const row = addUploadItem(file);
  selectPipeline("ingestion");
  const suffix = file.name.toLowerCase().split(".").pop();
  if (!["pdf", "txt"].includes(suffix)) {
    const error = new Error("Only PDF and TXT files are supported.");
    error.payload = { detail: { code: "unsupported_type" } };
    failIngestion(file, error, row);
    return;
  }
  if (file.size > maxUploadBytes) {
    const error = new Error(`File exceeds the ${formatBytes(maxUploadBytes)} upload limit.`);
    error.payload = { detail: { code: "upload_too_large" } };
    failIngestion(file, error, row);
    return;
  }

  setPipeline("ingestion", {
    state: "active",
    summary: "Request in progress. Internal stages remain unconfirmed until the API responds.",
    stages: ["active", "pending", "pending", "pending", "pending"],
    facts: ingestionFacts(file),
    scope: "POST /api/documents · awaiting response",
  });
  row.state.textContent = "Awaiting pipeline confirmation…";
  const formData = new FormData();
  formData.append("file", file);
  try {
    const result = await requestJson("/api/documents", { method: "POST", body: formData });
    const duplicate = result.status === "duplicate";
    row.item.classList.add("success");
    row.state.textContent = duplicate
      ? "Already indexed · no new vectors"
      : `${result.chunks} chunk${result.chunks === 1 ? "" : "s"} indexed`;
    setPipeline("ingestion", {
      state: "completed",
      summary: duplicate
        ? "Content hash matched an indexed document; vector creation was safely skipped."
        : "The API confirmed extraction, deterministic chunking, embeddings and persistent indexing.",
      stages: duplicate
        ? ["complete", "complete", "skipped", "skipped", "skipped"]
        : ["complete", "complete", "complete", "complete", "complete"],
      facts: ingestionFacts(file, {
        chunks: duplicate ? "0 new" : String(result.chunks),
        identity: `${result.document_id.slice(0, 12)}…`,
        result: duplicate ? "Duplicate" : "Indexed",
        confirmation: "API response confirmed",
      }),
      scope: "POST /api/documents · confirmed",
    });
    await loadDocuments();
  } catch (error) {
    failIngestion(file, error, row);
  }
}

async function handleFiles(files) {
  for (const file of [...files]) await uploadFile(file);
}

fileInput.addEventListener("change", () => {
  handleFiles(fileInput.files);
  fileInput.value = "";
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
});

dropZone.addEventListener("drop", (event) => handleFiles(event.dataTransfer.files));

function showDocumentMessage(message, isError = false) {
  documentMessage.textContent = message;
  documentMessage.className = `message${isError ? "" : " success"}`;
  documentMessage.hidden = false;
}

function describeDocument(document) {
  const locations = [];
  if (document.pages.length) locations.push(`pages ${document.pages.join(", ")}`);
  if (document.sections.length) locations.push(`sections ${document.sections.join(", ")}`);
  return [
    `${document.chunk_count} chunk${document.chunk_count === 1 ? "" : "s"}`,
    document.file_type.toUpperCase(),
    ...locations,
    `ID ${document.document_id.slice(0, 12)}…`,
  ].filter(Boolean).join(" · ");
}

function renderDocuments(documents) {
  documentCount.textContent = String(documents.length);
  heroDocuments.textContent = String(documents.length).padStart(2, "0");
  documentList.replaceChildren();
  if (documents.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "The archive is empty. Add a source to begin.";
    documentList.append(empty);
    return;
  }

  documents.forEach((indexedDocument) => {
    const item = document.createElement("div");
    item.className = "document-item";

    const summary = document.createElement("div");
    summary.className = "document-summary";
    const name = document.createElement("strong");
    name.className = "document-name";
    name.textContent = indexedDocument.filename;
    const metadata = document.createElement("p");
    metadata.className = "document-meta";
    metadata.textContent = describeDocument(indexedDocument);
    const source = document.createElement("p");
    source.className = "document-source";
    source.textContent = indexedDocument.source || "Source path unavailable";
    summary.append(name, metadata, source);

    const actions = document.createElement("div");
    actions.className = "document-actions";
    const replaceInput = document.createElement("input");
    replaceInput.className = "replacement-input";
    replaceInput.type = "file";
    replaceInput.accept = ".pdf,.txt,application/pdf,text/plain";
    replaceInput.setAttribute("aria-label", `Replacement file for ${indexedDocument.filename}`);
    const replaceButton = document.createElement("button");
    replaceButton.className = "document-action";
    replaceButton.type = "button";
    replaceButton.textContent = "Replace";
    replaceButton.setAttribute("aria-label", `Replace ${indexedDocument.filename}`);
    replaceButton.addEventListener("click", () => replaceInput.click());
    replaceInput.addEventListener("change", async () => {
      const file = replaceInput.files[0];
      replaceInput.value = "";
      if (!file) return;
      if (!window.confirm(`Replace ${indexedDocument.filename} with ${file.name}? The current index stays available if replacement fails.`)) return;
      selectPipeline("ingestion");
      setPipeline("ingestion", {
        state: "active",
        summary: "Replacement is in progress. The existing document remains indexed until success.",
        stages: ["active", "pending", "pending", "pending", "pending"],
        facts: ingestionFacts(file, { result: "Replacement pending" }),
        scope: "PUT /api/documents/{id} · awaiting response",
      });
      replaceButton.disabled = true;
      const formData = new FormData();
      formData.append("file", file);
      try {
        const result = await requestJson(`/api/documents/${encodeURIComponent(indexedDocument.document_id)}`, {
          method: "PUT",
          body: formData,
        });
        const duplicate = result.status === "duplicate";
        showDocumentMessage(duplicate
          ? `${indexedDocument.filename} is unchanged; identical content was already indexed.`
          : `${indexedDocument.filename} was replaced with ${result.filename}.`);
        setPipeline("ingestion", {
          state: "completed",
          summary: duplicate
            ? "Replacement matched the current content; no index mutation was performed."
            : "Replacement was indexed successfully before the previous document points were removed.",
          stages: duplicate
            ? ["complete", "complete", "skipped", "skipped", "skipped"]
            : ["complete", "complete", "complete", "complete", "complete"],
          facts: ingestionFacts(file, {
            chunks: duplicate ? "0 new" : String(result.chunks),
            identity: `${result.document_id.slice(0, 12)}…`,
            result: duplicate ? "Unchanged" : "Replaced",
            confirmation: "API response confirmed",
          }),
          scope: "PUT /api/documents/{id} · confirmed",
        });
        await loadDocuments();
      } catch (error) {
        failIngestion(file, error, { item: document.createElement("div"), state: document.createElement("span") });
        showDocumentMessage(`Replacement failed: ${error.message}`, true);
      } finally {
        replaceButton.disabled = false;
      }
    });

    const deleteButton = document.createElement("button");
    deleteButton.className = "document-action danger";
    deleteButton.type = "button";
    deleteButton.textContent = "Delete";
    deleteButton.setAttribute("aria-label", `Delete ${indexedDocument.filename}`);
    deleteButton.addEventListener("click", async () => {
      if (!window.confirm(`Delete ${indexedDocument.filename} from the index? This cannot be undone.`)) return;
      deleteButton.disabled = true;
      try {
        const result = await requestJson(`/api/documents/${encodeURIComponent(indexedDocument.document_id)}`, {
          method: "DELETE",
        });
        showDocumentMessage(`${indexedDocument.filename} deleted from the vector index (${result.deleted_chunks} chunks).`);
        await loadDocuments();
      } catch (error) {
        showDocumentMessage(`Deletion failed: ${error.message}`, true);
        deleteButton.disabled = false;
      }
    });

    actions.append(replaceInput, replaceButton, deleteButton);
    item.append(summary, actions);
    documentList.append(item);
  });
}

async function loadDocuments() {
  refreshDocuments.disabled = true;
  try {
    const result = await requestJson("/api/documents");
    renderDocuments(result.documents);
  } catch (error) {
    documentList.replaceChildren();
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "Could not read the local archive.";
    documentList.append(empty);
    showDocumentMessage(error.message, true);
  } finally {
    refreshDocuments.disabled = false;
  }
}

refreshDocuments.addEventListener("click", loadDocuments);

function showQuestionError(message) {
  questionMessage.textContent = message;
  questionMessage.hidden = false;
}

function addMetadata(container, label, value) {
  if (value === null || value === undefined || value === "") return;
  const entry = document.createElement("span");
  entry.textContent = `${label}: ${value}`;
  container.append(entry);
}

function qaStages(result) {
  if (result.status === "answered" || result.status === "unsupported") {
    return ["complete", "complete", "complete", "complete", "complete", "complete"];
  }
  if (result.status === "empty_index") {
    return ["complete", "skipped", "skipped", "skipped", "skipped", "complete"];
  }
  if (result.status === "model_unavailable") {
    return ["complete", "complete", "complete", "failed", "skipped", "skipped"];
  }
  if (result.status === "generation_error" && String(result.error || "").toLowerCase().includes("citation")) {
    return ["complete", "complete", "complete", "complete", "failed", "skipped"];
  }
  if (result.status === "generation_error") {
    return ["complete", "complete", "complete", "failed", "skipped", "skipped"];
  }
  return ["failed", "skipped", "skipped", "skipped", "skipped", "skipped"];
}

function updateQaFlow(result) {
  const trace = result.trace || {};
  const retrieved = Array.isArray(trace.retrieved_sources) ? trace.retrieved_sources : [];
  const sourceNames = [...new Set(retrieved.map((source) => source.document_title))];
  const scores = retrieved.map((source) => `${source.citation_id} ${Number(source.retrieval_score).toFixed(3)}`);
  const citations = Array.isArray(result.citation_ids) ? result.citation_ids : [];
  const failed = ["model_unavailable", "generation_error"].includes(result.status);
  const summary = {
    answered: "The API confirmed retrieval, bounded context, local generation and citation validation.",
    unsupported: "The model completed the grounded request and abstained because the evidence was insufficient.",
    empty_index: "Retrieval completed with no passages; generation was correctly skipped.",
    model_unavailable: "Evidence was prepared, but the configured local model was unavailable.",
    generation_error: "The answer pipeline stopped before a validated answer could be returned.",
  }[result.status] || "The question-answering request did not complete.";
  setPipeline("qa", {
    state: failed ? "failed" : "completed",
    summary,
    stages: qaStages(result),
    facts: {
      Model: trace.model || ollamaModel,
      Retrieved: `${retrieved.length} passage${retrieved.length === 1 ? "" : "s"}`,
      Sources: sourceNames.length ? sourceNames.join(" · ") : "None",
      Scores: scores.length ? scores.join(" · ") : "None",
      Context: `${trace.context_chars || 0} / ${trace.max_context_chars || maxContextChars} chars`,
      Citations: citations.length ? citations.join(", ") : "None returned",
    },
    scope: "POST /api/ask · response confirmed",
  });
}

function renderAnswer(result) {
  updateQaFlow(result);
  answerSection.hidden = false;
  answerText.textContent = result.answer;
  answerStatus.textContent = result.status.replaceAll("_", " ");
  answerStatus.className = `answer-status ${result.status}`;
  citationList.replaceChildren();

  const sources = Array.isArray(result.sources) ? result.sources : [];
  citationArea.hidden = sources.length === 0;
  citationCount.textContent = `${sources.length} citation${sources.length === 1 ? "" : "s"}`;
  sources.forEach((source) => {
    const details = document.createElement("details");
    details.className = "citation";
    const summary = document.createElement("summary");
    const id = document.createElement("span");
    id.className = "citation-id";
    id.textContent = source.citation_id;
    const title = document.createElement("span");
    title.textContent = source.document_title;
    summary.append(id, title);

    const metadata = document.createElement("div");
    metadata.className = "citation-details";
    addMetadata(metadata, "Page", source.page);
    addMetadata(metadata, "Section", source.section);
    addMetadata(metadata, "Chunk", source.chunk_index);
    addMetadata(metadata, "Similarity", Number(source.retrieval_score).toFixed(3));
    addMetadata(metadata, "Source", source.source);
    details.append(summary, metadata);
    citationList.append(details);
  });
  answerSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) {
    showQuestionError("Enter a question before asking the archive.");
    return;
  }

  selectPipeline("qa");
  setPipeline("qa", {
    state: "active",
    summary: "Request in progress. Only retrieval is active; downstream stages await the API response.",
    stages: ["active", "pending", "pending", "pending", "pending", "pending"],
    facts: {
      Model: ollamaModel,
      Retrieved: "Awaiting response",
      Sources: "Unconfirmed",
      Scores: "Unconfirmed",
      Context: `≤ ${maxContextChars} chars`,
      Citations: "Unconfirmed",
    },
    scope: "POST /api/ask · awaiting response",
  });
  questionMessage.hidden = true;
  askButton.disabled = true;
  askButton.firstElementChild.textContent = "Consulting archive…";
  try {
    const result = await requestJson("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, limit: 4 }),
    });
    renderAnswer(result);
  } catch (error) {
    const response = error.payload;
    if (response && response.answer) {
      renderAnswer(response);
    } else {
      setPipeline("qa", {
        state: "failed",
        summary: `The request failed before the backend confirmed retrieval: ${error.message}`,
        stages: ["failed", "skipped", "skipped", "skipped", "skipped", "skipped"],
        scope: "No completed API response",
      });
    }
    showQuestionError(error.message);
  } finally {
    askButton.disabled = false;
    askButton.firstElementChild.textContent = "Reveal answer";
  }
});

renderFlow();
checkHealth();
