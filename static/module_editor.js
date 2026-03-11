(function () {
  const moduleId = window.MODULE_ID;
  const initialModule = window.INITIAL_MODULE;
  if (!moduleId || !initialModule) return;

  const titleEl = document.getElementById("module-title");
  const overviewEl = document.getElementById("module-overview");
  const sectionsEl = document.getElementById("sections");
  const tocEl = document.getElementById("section-toc");
  const statusEl = document.getElementById("editor-status");
  const refreshBtn = document.getElementById("refresh-sources");
  const regenRefreshEl = document.getElementById("regen-refresh");
  const bannerEl = document.getElementById("module-summary-banner");
  const editorBannerEl = document.getElementById("editor-banner");
  const scrollTopEl = document.getElementById("scroll-top");

  let moduleState = initialModule;
  let sourceCountOverride = null;

  function setStatus(message, kind) {
    statusEl.textContent = message || "";
    statusEl.className = "status" + (kind ? " " + kind : "");
  }

  function showEditorBanner(message) {
    if (!editorBannerEl) return;
    editorBannerEl.textContent = message || "";
    editorBannerEl.hidden = !message;
  }

  function escapeHtml(input) {
    return String(input || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function sourceCount() {
    if (typeof sourceCountOverride === "number") {
      return sourceCountOverride;
    }
    const items = Array.isArray(moduleState.evidence_pack) ? moduleState.evidence_pack : [];
    return items.length;
  }

  function renderBanner() {
    const sections = Array.isArray(moduleState.sections) ? moduleState.sections : [];
    bannerEl.textContent = `This module has ${sections.length} sections and ${sourceCount()} sources.`;
  }

  function renderToc(sections) {
    if (!tocEl) return;
    tocEl.innerHTML = sections
      .map((section, idx) => {
        const sectionId = String(section.section_id || `section-${idx + 1}`);
        const heading = escapeHtml(section.heading || `Section ${idx + 1}`);
        return `<a href="#editor-${escapeHtml(sectionId)}">${heading}</a>`;
      })
      .join("");
  }

  function render() {
    titleEl.textContent = moduleState.title || "Untitled module";
    overviewEl.textContent = moduleState.overview || "";
    renderBanner();

    const sections = Array.isArray(moduleState.sections) ? moduleState.sections : [];
    renderToc(sections);
    sectionsEl.innerHTML = sections
      .map((section, idx) => {
        const citations = Array.isArray(section.citations) ? section.citations : [];
        const citationText = citations.length > 0 ? citations.join(", ") : "(none)";
        const sectionId = escapeHtml(section.section_id || `section-${idx + 1}`);
        return `
          <article class="section-card" id="editor-${sectionId}">
            <h3>${escapeHtml(section.heading)}</h3>
            <p class="section-meta">
              Objective #${escapeHtml(section.objective_index)} - ${escapeHtml(section.learning_goal || "")}
            </p>
            <div class="section-content">${escapeHtml(section.content || "")}</div>
            <p class="section-meta">Citations: ${escapeHtml(citationText)}</p>
            <label class="section-guidance-label" for="guidance-${sectionId}">
              What would you like to change?
            </label>
            <textarea
              id="guidance-${sectionId}"
              class="section-guidance"
              rows="2"
              placeholder="Example: simplify this section and add one concrete example."
            ></textarea>
            <button class="btn btn-emphasis regen-btn" data-section-id="${sectionId}" type="button">
              Improve this section
            </button>
          </article>
        `;
      })
      .join("");
  }

  async function parseError(response) {
    try {
      const payload = await response.json();
      return payload && payload.detail ? String(payload.detail) : "Request failed";
    } catch (_err) {
      return response.statusText || "Request failed";
    }
  }

  async function regenerateSection(sectionId, instructions) {
    setStatus(`Rewriting ${sectionId}...`, "");
    const response = await fetch(`/v1/modules/${moduleId}/sections/${sectionId}/regenerate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        instructions: instructions || null,
        refresh_sources: regenRefreshEl.checked,
      }),
    });

    if (!response.ok) {
      throw new Error(await parseError(response));
    }

    const payload = await response.json();
    moduleState = payload.module || moduleState;
    moduleState.evidence_pack = payload.evidence_pack || moduleState.evidence_pack || [];
    sourceCountOverride = null;
    render();
    setStatus(`Section ${sectionId} was rewritten.`, "success");
    showEditorBanner("Section improved successfully.");
  }

  async function refreshSources() {
    setStatus("Refreshing sources...", "");
    refreshBtn.disabled = true;
    const response = await fetch(`/v1/modules/${moduleId}/refresh_sources`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    refreshBtn.disabled = false;

    if (!response.ok) {
      throw new Error(await parseError(response));
    }

    const payload = await response.json();
    sourceCountOverride = Number(payload.evidence_count || 0);
    renderBanner();
    const message = `Sources refreshed: ${payload.evidence_count} total (${payload.doc_count} doc / ${payload.web_count} web).`;
    setStatus(message, "success");
    showEditorBanner("Sources refreshed successfully.");
  }

  sectionsEl.addEventListener("click", async (event) => {
    const button = event.target.closest(".regen-btn");
    if (!button) return;

    const sectionId = button.getAttribute("data-section-id");
    if (!sectionId) return;

    const card = button.closest(".section-card");
    const instructionsInput = card ? card.querySelector(".section-guidance") : null;
    const instructions = instructionsInput ? String(instructionsInput.value || "").trim() : "";

    const originalLabel = button.textContent;
    button.disabled = true;
    button.textContent = "Improving...";
    try {
      await regenerateSection(sectionId, instructions);
    } catch (error) {
      setStatus(error.message || "Failed to improve section.", "error");
      showEditorBanner("");
    }
    button.textContent = originalLabel;
    button.disabled = false;
  });

  refreshBtn.addEventListener("click", async () => {
    try {
      await refreshSources();
    } catch (error) {
      setStatus(error.message || "Failed to refresh sources.", "error");
      showEditorBanner("");
    }
  });

  if (scrollTopEl) {
    scrollTopEl.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  const params = new URLSearchParams(window.location.search);
  if (params.get("created") === "1") {
    showEditorBanner("Module created successfully.");
  }

  render();
})();
