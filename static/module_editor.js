(function () {
  const moduleId = window.MODULE_ID;
  const initialModule = window.INITIAL_MODULE;
  const initialHistory = window.INITIAL_HISTORY;
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
  const historyEl = document.getElementById("module-history");
  const shareBtn = document.getElementById("share-module");
  const copyShareBtn = document.getElementById("copy-share-link");
  const shareBadgeEl = document.getElementById("share-badge");
  const sharePanelEl = document.getElementById("share-panel");
  const shareLinkEl = document.getElementById("share-link");
  const openShareLinkEl = document.getElementById("open-share-link");
  const tutorQuestionEl = document.getElementById("tutor-question");
  const tutorAskBtn = document.getElementById("tutor-ask");
  const tutorHistoryEl = document.getElementById("tutor-history");
  const tutorSuggestionsEl = document.getElementById("tutor-suggestions");
  const tutorModesEl = document.getElementById("tutor-modes");
  const tutorLabelEl = document.querySelector('label[for="tutor-question"]');
  const quizPromptCardEl = document.getElementById("quiz-prompt-card");
  const quizPromptTextEl = document.getElementById("quiz-prompt-text");
  const assignmentTopBtn = document.getElementById("create-assignment");
  const assignmentPanelEl = document.getElementById("assignment-panel");
  const assignmentGenerateBtn = document.getElementById("assignment-generate");
  const assignmentStatusEl = document.getElementById("assignment-status");
  const assignmentEmptyEl = document.getElementById("assignment-empty");
  const assignmentOutputEl = document.getElementById("assignment-output");
  const assignmentPromptEl = document.getElementById("assignment-prompt");
  const assignmentRubricEl = document.getElementById("assignment-rubric");
  const assignmentExportMarkdownBtn = document.getElementById("assignment-export-markdown");
  const assignmentExportJsonBtn = document.getElementById("assignment-export-json");
  const studentResponseEl = document.getElementById("student-response");
  const assignmentGradeBtn = document.getElementById("assignment-grade");
  const assignmentGradeOutputEl = document.getElementById("assignment-grade-output");
  const assignmentScoreEl = document.getElementById("assignment-score");
  const assignmentFeedbackEl = document.getElementById("assignment-feedback");
  const assignmentBreakdownEl = document.getElementById("assignment-breakdown");
  const assignmentGradeBadgeEl = document.getElementById("assignment-grade-badge");

  let moduleState = initialModule;
  let historyState = Array.isArray(initialHistory) ? initialHistory : [];
  let sourceCountOverride = null;
  let highlightResetTimer = null;
  const sectionDiffState = {};
  const guidanceDrafts = {};
  const evidenceCache = {};
  const tutorHistory = [];
  let activeQuizPrompt = "";
  let assignmentState = null;
  let gradeState = null;

  function setStatus(message, kind) {
    statusEl.textContent = message || "";
    statusEl.className = "status" + (kind ? " " + kind : "");
  }

  function showEditorBanner(message, kind) {
    if (!editorBannerEl) return;
    editorBannerEl.textContent = message || "";
    editorBannerEl.className = "alert" + (kind ? " " + kind : " info");
    editorBannerEl.hidden = !message;
  }

  function setAssignmentStatus(message, kind) {
    if (!assignmentStatusEl) return;
    assignmentStatusEl.textContent = message || "";
    assignmentStatusEl.className = "status" + (kind ? " " + kind : "");
  }

  function setButtonBusy(button, isBusy, busyLabel) {
    if (!button) return;
    if (isBusy) {
      button.dataset.originalLabel = button.textContent;
      button.disabled = true;
      button.classList.add("is-loading");
      button.textContent = busyLabel;
      return;
    }
    button.disabled = false;
    button.classList.remove("is-loading");
    button.textContent = button.dataset.originalLabel || button.textContent;
  }

  function friendlyEditorMessage(context, detail) {
    const text = String(detail || "").toLowerCase();
    if (text.includes("took too long") || text.includes("timed out") || text.includes("504")) {
      return "That step took too long. Please try again.";
    }
    if (text.includes("missing openai_api_key")) {
      return "The app is missing an OpenAI API key. Add it to .env and restart the server.";
    }
    if (context === "tutor") {
      return "We couldn't answer that question just now. Please try again.";
    }
    if (context === "assignment") {
      return "We couldn't create the assignment right now. Please try again.";
    }
    if (context === "grading") {
      return "We couldn't grade that response right now. Please try again.";
    }
    if (context === "export") {
      return "We couldn't export the assignment just now. Please try again.";
    }
    if (context === "share") {
      return "We couldn't update sharing right now. Please try again.";
    }
    if (context === "refresh") {
      return "We couldn't refresh the sources right now. Please try again.";
    }
    if (context === "revert") {
      return "We couldn't restore that version right now. Please try again.";
    }
    return "Something went wrong. Please try again.";
  }

  function normalizedCitations(citations) {
    return (Array.isArray(citations) ? citations : [])
      .map((item) => String(item || "").trim())
      .filter(Boolean);
  }

  function citationsChanged(beforeCitations, afterCitations) {
    const before = normalizedCitations(beforeCitations);
    const after = normalizedCitations(afterCitations);
    return before.join("|") !== after.join("|");
  }

  function sectionSnapshot(section) {
    return {
      content: String((section && section.content) || ""),
      citations: normalizedCitations(section && section.citations),
    };
  }

  function escapeHtml(input) {
    return String(input || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function escapeAttribute(input) {
    return escapeHtml(input);
  }

  function sourceCount() {
    if (typeof sourceCountOverride === "number") {
      return sourceCountOverride;
    }
    const items = Array.isArray(moduleState.evidence_pack) ? moduleState.evidence_pack : [];
    return items.length;
  }

  function currentShareUrl() {
    if (moduleState.share_url) {
      return String(moduleState.share_url);
    }
    const shareId = String(moduleState.share_id || "").trim();
    if (!shareId) {
      return "";
    }
    return `${window.location.origin}/shared/${encodeURIComponent(shareId)}`;
  }

  async function copyText(value) {
    if (!value) return;
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(value);
      return;
    }

    const helper = document.createElement("textarea");
    helper.value = value;
    helper.setAttribute("readonly", "");
    helper.style.position = "absolute";
    helper.style.left = "-9999px";
    document.body.appendChild(helper);
    helper.select();
    document.execCommand("copy");
    document.body.removeChild(helper);
  }

  function slugifyFilenamePart(value, fallback) {
    const cleaned = String(value || "")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "");
    return cleaned || fallback;
  }

  function assignmentFilename(extension) {
    const moduleSlug = slugifyFilenamePart(moduleState.title || "assignment", "assignment");
    return `${moduleSlug}-assignment.${extension}`;
  }

  function triggerDownload(filename, content, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.setTimeout(function () {
      URL.revokeObjectURL(url);
    }, 0);
  }

  function renderBanner() {
    const sections = Array.isArray(moduleState.sections) ? moduleState.sections : [];
    bannerEl.textContent = `This module has ${sections.length} sections and ${sourceCount()} sources.`;
  }

  function renderShareControls() {
    if (!shareBtn || !copyShareBtn || !sharePanelEl || !shareLinkEl || !openShareLinkEl) return;

    const shareEnabled = Boolean(moduleState.share_enabled);
    const shareUrl = currentShareUrl();

    shareBtn.textContent = shareEnabled ? "Disable sharing" : "Share module";
    if (shareBadgeEl) {
      shareBadgeEl.hidden = !shareEnabled;
    }
    sharePanelEl.hidden = !shareEnabled;
    copyShareBtn.hidden = !shareEnabled;
    shareLinkEl.value = shareEnabled ? shareUrl : "";
    openShareLinkEl.href = shareEnabled ? shareUrl : "#";
  }

  function clearAssignmentState() {
    assignmentState = null;
    gradeState = null;
    if (studentResponseEl) {
      studentResponseEl.value = "";
    }
  }

  function renderRubricLevel(level) {
    const score = escapeHtml(level && level.score);
    const description = escapeHtml(level && level.description);
    return `
      <li class="rubric-level">
        <span class="rubric-score">${score}</span>
        <span>${description}</span>
      </li>
    `;
  }

  function renderRubricCriterion(item) {
    const levels = Array.isArray(item && item.levels) ? item.levels : [];
    return `
      <article class="rubric-criterion">
        <h4>${escapeHtml(item && item.criteria)}</h4>
        <ul class="rubric-levels">
          ${levels.map(renderRubricLevel).join("")}
        </ul>
      </article>
    `;
  }

  function renderGradeBreakdown(items) {
    const breakdown = Array.isArray(items) ? items : [];
    if (!assignmentBreakdownEl) return;
    if (!breakdown.length) {
      assignmentBreakdownEl.innerHTML = '<p class="helper-text">No criterion-by-criterion breakdown was available.</p>';
      return;
    }
    assignmentBreakdownEl.innerHTML = breakdown.map(function (item) {
      return `
        <article class="grade-breakdown-item">
          <div class="grade-breakdown-head">
            <strong>${escapeHtml(item.criteria)}</strong>
            <span>${escapeHtml(item.score)}/${escapeHtml(item.max_score)}</span>
          </div>
          <p>${escapeHtml(item.feedback)}</p>
        </article>
      `;
    }).join("");
  }

  function renderAssignment() {
    if (!assignmentOutputEl || !assignmentPromptEl || !assignmentRubricEl || !assignmentGradeOutputEl) return;
    if (!assignmentState) {
      if (assignmentEmptyEl) {
        assignmentEmptyEl.hidden = false;
      }
      assignmentOutputEl.hidden = true;
      assignmentGradeOutputEl.hidden = true;
      return;
    }

    if (assignmentEmptyEl) {
      assignmentEmptyEl.hidden = true;
    }
    assignmentOutputEl.hidden = false;
    assignmentPromptEl.textContent = assignmentState.prompt || "";
    assignmentRubricEl.innerHTML = (Array.isArray(assignmentState.rubric) ? assignmentState.rubric : [])
      .map(renderRubricCriterion)
      .join("");

    if (!gradeState) {
      assignmentGradeOutputEl.hidden = true;
      return;
    }

    assignmentGradeOutputEl.hidden = false;
    if (assignmentScoreEl) {
      assignmentScoreEl.textContent = `${gradeState.score}/100`;
    }
    if (assignmentFeedbackEl) {
      assignmentFeedbackEl.textContent = gradeState.feedback || "";
    }
    if (assignmentGradeBadgeEl) {
      assignmentGradeBadgeEl.textContent = gradeState.unverified ? "Unverified" : "Grounded";
      assignmentGradeBadgeEl.className = gradeState.unverified
        ? "tutor-badge tutor-badge-unverified"
        : "tutor-badge tutor-badge-grounded";
    }
    renderGradeBreakdown(gradeState.breakdown);
  }

  function formatVersionTimestamp(value) {
    const date = value ? new Date(value) : null;
    if (!date || Number.isNaN(date.getTime())) {
      return "Unknown time";
    }
    return date.toLocaleString([], {
      dateStyle: "medium",
      timeStyle: "short",
    });
  }

  function formatVersionAction(action) {
    if (action === "section_improved") return "Section improved";
    if (action === "sources_refreshed") return "Sources refreshed";
    if (action === "generated") return "Generated";
    return "Updated";
  }

  function renderHistory() {
    if (!historyEl) return;
    if (!historyState.length) {
      historyEl.innerHTML = '<p class="helper-text">No versions yet.</p>';
      return;
    }

    historyEl.innerHTML = historyState
      .map(function (entry) {
        const versionId = escapeHtml(entry.version_id || "");
        const actionLabel = escapeHtml(formatVersionAction(entry.action));
        const timestamp = escapeHtml(formatVersionTimestamp(entry.timestamp));
        return `
          <article class="history-item">
            <div class="history-copy">
              <strong>${actionLabel}</strong>
              <span>${timestamp}</span>
            </div>
            <button
              class="btn btn-secondary btn-small history-revert-btn"
              data-version-id="${versionId}"
              type="button"
            >
              Revert to this version
            </button>
          </article>
        `;
      })
      .join("");
  }

  function currentTutorMode() {
    const selected = tutorModesEl
      ? tutorModesEl.querySelector('input[name="tutor-mode"]:checked')
      : null;
    return selected ? String(selected.value || "default") : "default";
  }

  function setTutorMode(mode) {
    if (!tutorModesEl) return;
    const target = tutorModesEl.querySelector(`input[name="tutor-mode"][value="${mode}"]`);
    if (target) {
      target.checked = true;
    }
    updateTutorModeUI();
  }

  function suggestedQuestions() {
    const sections = Array.isArray(moduleState.sections) ? moduleState.sections : [];
    const topicLabel = String(moduleState.title || "this module").replace(/\s+module$/i, "").trim() || "this topic";
    const firstHeading = String((sections[0] && sections[0].heading) || topicLabel).trim();
    const secondHeading = String((sections[1] && sections[1].heading) || firstHeading).trim();
    const suggestions = [
      `Can you explain ${topicLabel} more simply?`,
      `What is the most important idea in ${firstHeading}?`,
      `Can you give me a real-life example of ${secondHeading}?`,
    ];
    return suggestions.filter(function (item, index) {
      return item && suggestions.indexOf(item) === index;
    }).slice(0, 3);
  }

  function renderSuggestedQuestions() {
    if (!tutorSuggestionsEl) return;
    tutorSuggestionsEl.innerHTML = suggestedQuestions()
      .map(function (question) {
        return `
          <button
            class="btn btn-secondary btn-small suggestion-btn"
            data-question="${escapeAttribute(question)}"
            type="button"
          >
            ${escapeHtml(question)}
          </button>
        `;
      })
      .join("");
  }

  function updateTutorModeUI() {
    const mode = currentTutorMode();
    const hasActiveQuiz = Boolean(activeQuizPrompt);

    if (quizPromptCardEl && quizPromptTextEl) {
      quizPromptCardEl.hidden = !(hasActiveQuiz && mode === "quiz_me");
      quizPromptTextEl.textContent = activeQuizPrompt || "";
    }

    if (tutorLabelEl) {
      tutorLabelEl.textContent = hasActiveQuiz && mode === "quiz_me"
        ? "Your answer"
        : "Ask a question about this module";
    }

    if (tutorQuestionEl) {
      if (mode === "quiz_me" && hasActiveQuiz) {
        tutorQuestionEl.placeholder = "Type your answer to the quiz question here.";
      } else if (mode === "quiz_me") {
        tutorQuestionEl.placeholder = "Leave this blank and click the button to get a quiz question.";
      } else if (mode === "simpler") {
        tutorQuestionEl.placeholder = "Example: Can you explain chlorophyll in simpler words?";
      } else if (mode === "more_detailed") {
        tutorQuestionEl.placeholder = "Example: Can you explain the process step by step?";
      } else {
        tutorQuestionEl.placeholder = "Example: What role does chlorophyll play in photosynthesis?";
      }
    }

    if (tutorAskBtn) {
      if (mode === "quiz_me" && hasActiveQuiz) {
        tutorAskBtn.textContent = "Check my answer";
      } else if (mode === "quiz_me") {
        tutorAskBtn.textContent = "Get quiz question";
      } else {
        tutorAskBtn.textContent = "Ask the tutor";
      }
    }
  }

  function renderTutorHistory() {
    if (!tutorHistoryEl) return;
    if (!tutorHistory.length) {
      tutorHistoryEl.innerHTML = [
        '<div class="empty-subpanel">',
        '<p class="empty-title">No tutor history yet.</p>',
        '<p class="helper-text">Ask a question to get a grounded explanation with citations from this module.</p>',
        "</div>",
      ].join("");
      return;
    }

    tutorHistoryEl.innerHTML = tutorHistory
      .slice()
      .reverse()
      .map(function (entry) {
        const citations = Array.isArray(entry.citations) ? entry.citations : [];
        const citationsMarkup = citations.length
          ? `
            <details class="tutor-citations">
              <summary>Sources (${citations.length})</summary>
              <ul class="citation-list tutor-citation-list">
                ${citations.map(renderCitationCard).join("")}
              </ul>
            </details>
          `
          : '<p class="section-meta">Sources: none available for this answer.</p>';
        const badge = entry.unverified
          ? '<span class="tutor-badge tutor-badge-unverified">Unverified</span>'
          : '<span class="tutor-badge tutor-badge-grounded">Grounded</span>';
        const cautionMarkup = entry.unverified
          ? '<p class="section-change-note tutor-unverified">This answer is cautious because the saved sources do not fully support it.</p>'
          : "";
        const promptMarkup = entry.kind === "quiz_question"
          ? '<p class="tutor-entry-label">Quiz question</p>'
          : entry.kind === "quiz_feedback"
            ? `
              <p class="tutor-entry-label">Quiz feedback</p>
              <p class="tutor-quiz-prompt"><strong>Question:</strong> ${escapeHtml(entry.quizPrompt || "")}</p>
              <p class="tutor-quiz-answer"><strong>Your answer:</strong> ${escapeHtml(entry.question || "")}</p>
            `
            : `<p class="tutor-entry-label">Question</p><p class="tutor-question">${escapeHtml(entry.question || "")}</p>`;
        return `
          <article class="tutor-answer-card">
            <div class="tutor-answer-header">
              ${badge}
            </div>
            ${promptMarkup}
            <div class="tutor-answer">${escapeHtml(entry.answer)}</div>
            ${cautionMarkup}
            <div class="tutor-sources">
              <h4>Sources</h4>
              ${citationsMarkup}
            </div>
          </article>
        `;
      })
      .join("");
  }

  function syncEvidenceCache(items) {
    (Array.isArray(items) ? items : []).forEach(function (item) {
      if (!item || !item.evidence_id) return;
      evidenceCache[String(item.evidence_id)] = item;
    });
  }

  function evidenceForCitation(citationId) {
    return evidenceCache[String(citationId || "").trim()] || null;
  }

  function renderCitationCard(citationId) {
    const citationKey = String(citationId || "").trim();
    const evidence = evidenceForCitation(citationKey);
    if (!citationKey) {
      return "";
    }
    if (!evidence) {
      return `
        <li>
          <div class="citation-card citation-missing">
            <span class="citation-kicker">Citation</span>
            <strong>${escapeHtml(citationKey)}</strong>
            <span class="citation-snippet">Source details are not currently available for this citation.</span>
          </div>
        </li>
      `;
    }

    const title = escapeHtml(evidence.title || evidence.doc_name || citationKey);
    const snippet = escapeHtml(evidence.snippet || "");
    if (String(evidence.source_type || "") === "web" && evidence.url) {
      const domain = escapeHtml(evidence.domain || evidence.url);
      return `
        <li>
          <a
            class="citation-card citation-link"
            href="${escapeAttribute(evidence.url)}"
            target="_blank"
            rel="noopener noreferrer"
          >
            <span class="citation-kicker">${domain}</span>
            <strong>${title}</strong>
            <span class="citation-open">Open source</span>
            <span class="citation-snippet">${snippet}</span>
          </a>
        </li>
      `;
    }

    if (String(evidence.source_type || "") === "web") {
      const domain = escapeHtml(evidence.domain || "Web source");
      return `
        <li>
          <div class="citation-card citation-missing">
            <span class="citation-kicker">${domain}</span>
            <strong>${title}</strong>
            <span class="citation-open citation-open-muted">URL unavailable</span>
            <span class="citation-snippet">${snippet}</span>
          </div>
        </li>
      `;
    }

    const docName = escapeHtml(evidence.doc_name || evidence.title || "Document source");
    const location = escapeHtml(evidence.location || "Location not available");
    return `
      <li>
        <details class="citation-card citation-doc-card">
          <summary>
            <span class="citation-kicker">Document source</span>
            <strong>${docName}</strong>
            <span class="citation-subline">${location}</span>
          </summary>
          <span class="citation-snippet">${snippet}</span>
        </details>
      </li>
    `;
  }

  function renderSectionCitations(section) {
    const citations = Array.isArray(section.citations) ? section.citations : [];
    if (!citations.length) {
      return '<p class="section-meta">Sources: none attached to this section.</p>';
    }

    return `
      <section class="section-citations">
        <h4>Sources</h4>
        <ul class="citation-list">
          ${citations.map(renderCitationCard).join("")}
        </ul>
      </section>
    `;
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

  function scrollToUpdatedSection(sectionId) {
    const sectionEl = document.getElementById(`editor-${sectionId}`);
    if (!sectionEl) return;

    sectionEl.scrollIntoView({ behavior: "smooth", block: "start" });
    sectionEl.classList.add("section-card-updated");
    if (highlightResetTimer) {
      window.clearTimeout(highlightResetTimer);
    }
    highlightResetTimer = window.setTimeout(function () {
      sectionEl.classList.remove("section-card-updated");
    }, 2200);
  }

  function render() {
    titleEl.textContent = moduleState.title || "Untitled module";
    overviewEl.textContent = moduleState.overview || "";
    renderBanner();
    renderShareControls();
    renderAssignment();
    syncEvidenceCache(moduleState.evidence_pack);
    renderSuggestedQuestions();
    renderHistory();
    renderTutorHistory();
    updateTutorModeUI();

    const sections = Array.isArray(moduleState.sections) ? moduleState.sections : [];
    renderToc(sections);
    sectionsEl.innerHTML = sections
      .map((section, idx) => {
        const sectionId = escapeHtml(section.section_id || `section-${idx + 1}`);
        const rawSectionId = section.section_id || `section-${idx + 1}`;
        const diffState = sectionDiffState[rawSectionId];
        const showChangesToggle = Boolean(diffState && diffState.contentChanged);
        const showChangesPanel = Boolean(diffState && diffState.showChanges);
        const guidanceValue = escapeHtml(guidanceDrafts[rawSectionId] || "");
        const changesToggle = showChangesToggle
          ? `
            <button
              class="btn btn-secondary btn-small change-toggle"
              data-section-id="${sectionId}"
              type="button"
            >
              ${diffState.showChanges ? "Hide changes" : "View changes"}
            </button>
          `
          : "";
        const changesPanel = showChangesPanel
          ? `
            <div class="change-panel">
              <div class="change-column">
                <p class="change-label">Previous content</p>
                <div class="change-content">${escapeHtml(diffState.previousContent)}</div>
              </div>
              <div class="change-column">
                <p class="change-label">Updated content</p>
                <div class="change-content">${escapeHtml(diffState.updatedContent)}</div>
              </div>
            </div>
          `
          : "";
        const citationsUpdatedNote = diffState && diffState.citationsChanged
          ? `<p class="section-change-note">Sources updated.</p>`
          : "";
        return `
          <article class="section-card" id="editor-${sectionId}">
            <h3>${escapeHtml(section.heading)}</h3>
            <p class="section-meta">
              Objective #${escapeHtml(section.objective_index)} - ${escapeHtml(section.learning_goal || "")}
            </p>
            <div class="section-content">${escapeHtml(section.content || "")}</div>
            ${renderSectionCitations(section)}
            ${citationsUpdatedNote}
            <label class="section-guidance-label" for="guidance-${sectionId}">
              What would you like to change?
            </label>
            <textarea
              id="guidance-${sectionId}"
              class="section-guidance"
              data-section-id="${sectionId}"
              rows="2"
              placeholder="Example: simplify this section and add one concrete example."
            >${guidanceValue}</textarea>
            <div class="section-actions">
              <button class="btn btn-emphasis regen-btn" data-section-id="${sectionId}" type="button">
                Improve this section
              </button>
              ${changesToggle}
            </div>
            ${changesPanel}
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

  async function loadHistory() {
    const response = await fetch(`/v1/modules/${moduleId}/history`);
    if (!response.ok) {
      throw new Error(await parseError(response));
    }
    historyState = await response.json();
    renderHistory();
  }

  async function requestAssignment() {
    const response = await fetch(`/v1/modules/${moduleId}/assignment`, {
      method: "POST",
    });
    if (!response.ok) {
      throw new Error(await parseError(response));
    }
    return response.json();
  }

  async function requestGrade(studentResponse, rubric) {
    const response = await fetch(`/v1/modules/${moduleId}/grade`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        student_response: studentResponse,
        rubric: rubric,
      }),
    });
    if (!response.ok) {
      throw new Error(await parseError(response));
    }
    return response.json();
  }

  async function requestAssignmentMarkdownExport(assignment) {
    const response = await fetch(`/v1/modules/${moduleId}/assignment/export/markdown`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(assignment),
    });
    if (!response.ok) {
      throw new Error(await parseError(response));
    }
    return response.text();
  }

  async function requestAssignmentJsonExport(assignment) {
    const response = await fetch(`/v1/modules/${moduleId}/assignment/export/json`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(assignment),
    });
    if (!response.ok) {
      throw new Error(await parseError(response));
    }
    return response.json();
  }

  async function toggleShare(button) {
    const enabling = !moduleState.share_enabled;
    setStatus(enabling ? "Turning on sharing..." : "Turning off sharing...", "loading");
    showEditorBanner(enabling ? "Turning sharing on..." : "Turning sharing off...", "info");
    setButtonBusy(button, true, enabling ? "Sharing..." : "Turning off...");
    setButtonBusy(copyShareBtn, true, "Working...");

    try {
      const response = await fetch(`/v1/modules/${moduleId}/share`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: enabling }),
      });
      if (!response.ok) {
        throw new Error(await parseError(response));
      }

      const payload = await response.json();
      moduleState.share_enabled = Boolean(payload.share_enabled);
      moduleState.share_id = payload.share_id || null;
      moduleState.share_url = payload.share_url || null;
      renderShareControls();
      if (moduleState.share_enabled) {
        setStatus("Sharing enabled.", "success");
        showEditorBanner("Sharing enabled. Your public link is ready.", "success");
      } else {
        setStatus("Sharing disabled. Existing shared links will no longer work.", "success");
        showEditorBanner("Sharing disabled. Existing shared links will no longer work.", "success");
      }
    } finally {
      setButtonBusy(button, false);
      setButtonBusy(copyShareBtn, false);
    }
  }

  async function generateAssignment() {
    setAssignmentStatus("Creating assignment...", "loading");
    showEditorBanner("Creating an assignment and rubric from the current module...", "info");
    const payload = await requestAssignment();
    assignmentState = payload;
    gradeState = null;
    if (studentResponseEl) {
      studentResponseEl.value = "";
    }
    renderAssignment();
    setAssignmentStatus("Assignment ready.", "success");
    showEditorBanner("Assignment created successfully.", "success");
    if (assignmentPanelEl) {
      assignmentPanelEl.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  async function gradeAssignment() {
    const responseText = studentResponseEl ? String(studentResponseEl.value || "").trim() : "";
    if (!assignmentState || !Array.isArray(assignmentState.rubric) || !assignmentState.rubric.length) {
      setAssignmentStatus("Create an assignment before grading a response.", "error");
      return;
    }
    if (!responseText) {
      setAssignmentStatus("Please enter a student response to grade.", "error");
      return;
    }
    setAssignmentStatus("Grading response...", "loading");
    showEditorBanner("Grading the response against the rubric...", "info");
    gradeState = await requestGrade(responseText, assignmentState.rubric);
    renderAssignment();
    setAssignmentStatus(
      gradeState.unverified ? "Grade is cautious because support was limited." : "Grade ready.",
      gradeState.unverified ? "" : "success",
    );
    showEditorBanner(
      gradeState.unverified
        ? "Response graded cautiously because support was limited."
        : "Response graded successfully.",
      gradeState.unverified ? "info" : "success",
    );
  }

  async function exportAssignmentMarkdown() {
    if (!assignmentState) {
      setAssignmentStatus("Create an assignment before exporting it.", "error");
      return;
    }
    setAssignmentStatus("Preparing Markdown export...", "loading");
    showEditorBanner("Preparing the assignment Markdown export...", "info");
    const markdown = await requestAssignmentMarkdownExport(assignmentState);
    triggerDownload(assignmentFilename("md"), markdown, "text/markdown;charset=utf-8");
    setAssignmentStatus("Assignment Markdown exported.", "success");
    showEditorBanner("Assignment exported successfully.", "success");
  }

  async function exportAssignmentJson() {
    if (!assignmentState) {
      setAssignmentStatus("Create an assignment before exporting it.", "error");
      return;
    }
    setAssignmentStatus("Preparing JSON export...", "loading");
    showEditorBanner("Preparing the assignment JSON export...", "info");
    const payload = await requestAssignmentJsonExport(assignmentState);
    triggerDownload(
      assignmentFilename("json"),
      `${JSON.stringify(payload, null, 2)}\n`,
      "application/json;charset=utf-8",
    );
    setAssignmentStatus("Assignment JSON exported.", "success");
    showEditorBanner("Assignment exported successfully.", "success");
  }

  async function askTutorQuestion(payload) {
    const response = await fetch(`/v1/modules/${moduleId}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      throw new Error(await parseError(response));
    }
    return response.json();
  }

  async function submitTutorRequest(options) {
    const mode = (options && options.mode) || currentTutorMode();
    const text = (options && options.question) != null
      ? String(options.question || "").trim()
      : String((tutorQuestionEl && tutorQuestionEl.value) || "").trim();
    const currentQuizPrompt = activeQuizPrompt;

    if (mode === "quiz_me" && currentQuizPrompt && !text) {
      setStatus("Please answer the quiz question before checking your answer.", "error");
      return;
    }
    if (mode !== "quiz_me" && !text) {
      setStatus("Please enter a question about this module.", "error");
      return;
    }

    const payload = {
      question: mode === "quiz_me" && !currentQuizPrompt ? "" : text,
      mode: mode,
    };
    if (mode === "quiz_me" && currentQuizPrompt) {
      payload.quiz_prompt = currentQuizPrompt;
    }

    setStatus(mode === "quiz_me" ? "Working on your quiz..." : "Answering your question...", "loading");
    showEditorBanner(
      mode === "quiz_me"
        ? "Tutor is preparing the next quiz step from the saved sources..."
        : "Tutor is reading the module and drafting an answer...",
      "info",
    );
    const response = await askTutorQuestion(payload);

    if (mode === "quiz_me" && !currentQuizPrompt) {
      if (!response.unverified) {
        activeQuizPrompt = String(response.answer || "");
      }
      tutorHistory.push({
        kind: "quiz_question",
        question: "",
        answer: String(response.answer || ""),
        citations: Array.isArray(response.citations) ? response.citations : [],
        unverified: Boolean(response.unverified),
      });
      if (tutorQuestionEl) {
        tutorQuestionEl.value = "";
      }
      renderTutorHistory();
      updateTutorModeUI();
      setStatus(response.unverified ? "Quiz question could not be fully verified." : "Quiz question ready.", response.unverified ? "" : "success");
      showEditorBanner(
        response.unverified ? "Quiz question is cautious because support was limited." : "Quiz question ready.",
        response.unverified ? "info" : "success",
      );
      return;
    }

    if (mode === "quiz_me") {
      tutorHistory.push({
        kind: "quiz_feedback",
        question: text,
        quizPrompt: currentQuizPrompt,
        answer: String(response.answer || ""),
        citations: Array.isArray(response.citations) ? response.citations : [],
        unverified: Boolean(response.unverified),
      });
      activeQuizPrompt = "";
      if (tutorQuestionEl) {
        tutorQuestionEl.value = "";
      }
      renderTutorHistory();
      updateTutorModeUI();
      setStatus(response.unverified ? "Feedback is cautious because support was limited." : "Quiz feedback ready.", response.unverified ? "" : "success");
      showEditorBanner(
        response.unverified ? "Quiz feedback is cautious because support was limited." : "Quiz feedback ready.",
        response.unverified ? "info" : "success",
      );
      return;
    }

    tutorHistory.push({
      kind: "answer",
      question: text,
      answer: String(response.answer || ""),
      citations: Array.isArray(response.citations) ? response.citations : [],
      unverified: Boolean(response.unverified),
    });
    if (tutorQuestionEl) {
      tutorQuestionEl.value = "";
    }
    renderTutorHistory();
    setStatus(response.unverified ? "Answered cautiously from the saved sources." : "Answer ready.", response.unverified ? "" : "success");
    showEditorBanner(
      response.unverified ? "Tutor answer is cautious because support was limited." : "Tutor answer ready.",
      response.unverified ? "info" : "success",
    );
  }

  async function revertToVersion(versionId, button) {
    setStatus("Restoring this version...", "loading");
    showEditorBanner("Restoring this saved version...", "info");
    setButtonBusy(button, true, "Restoring...");

    try {
      const response = await fetch(`/v1/modules/${moduleId}/revert/${versionId}`, {
        method: "POST",
      });
      if (!response.ok) {
        throw new Error(await parseError(response));
      }

      moduleState = await response.json();
      moduleState.evidence_pack = Array.isArray(moduleState.evidence_pack) ? moduleState.evidence_pack : [];
      syncEvidenceCache(moduleState.evidence_pack);
      sourceCountOverride = null;
      activeQuizPrompt = "";
      clearAssignmentState();
      await loadHistory();
      render();
      showEditorBanner("Module restored successfully.", "success");
      setStatus("Version restored.", "success");
      window.scrollTo({ top: 0, behavior: "smooth" });
    } finally {
      setButtonBusy(button, false);
    }
  }

  async function regenerateSection(sectionId, instructions) {
    const previousSection = (Array.isArray(moduleState.sections) ? moduleState.sections : []).find(function (section) {
      return String(section.section_id || "") === String(sectionId);
    });
    const previousSnapshot = sectionSnapshot(previousSection);

    setStatus(`Improving ${sectionId}...`, "loading");
    showEditorBanner("Improving this section from the saved sources...", "info");
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
    syncEvidenceCache(moduleState.evidence_pack);
    sourceCountOverride = null;
    activeQuizPrompt = "";
    clearAssignmentState();
    const updatedSection = (Array.isArray(moduleState.sections) ? moduleState.sections : []).find(function (section) {
      return String(section.section_id || "") === String(sectionId);
    });
    const updatedSnapshot = sectionSnapshot(updatedSection);
    sectionDiffState[sectionId] = {
      previousContent: previousSnapshot.content,
      updatedContent: updatedSnapshot.content,
      contentChanged: previousSnapshot.content !== updatedSnapshot.content,
      citationsChanged: citationsChanged(previousSnapshot.citations, updatedSnapshot.citations),
      showChanges: false,
    };
    guidanceDrafts[sectionId] = "";
    render();
    await loadHistory();
    window.requestAnimationFrame(function () {
      scrollToUpdatedSection(sectionId);
    });
    showEditorBanner("Section updated successfully.", "success");
    if (sectionDiffState[sectionId].citationsChanged) {
      setStatus("Sources updated.", "success");
    } else {
      setStatus("Section updated successfully.", "success");
    }
  }

  async function refreshSources() {
    setStatus("Refreshing sources...", "loading");
    showEditorBanner("Refreshing the saved sources for this module...", "info");
    setButtonBusy(refreshBtn, true, "Refreshing...");
    let payload;
    try {
      const response = await fetch(`/v1/modules/${moduleId}/refresh_sources`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!response.ok) {
        throw new Error(await parseError(response));
      }
      payload = await response.json();
    } finally {
      setButtonBusy(refreshBtn, false);
    }
    sourceCountOverride = Number(payload.evidence_count || 0);
    activeQuizPrompt = "";
    clearAssignmentState();
    await loadHistory();
    render();
    const message = `Sources refreshed: ${payload.evidence_count} total (${payload.doc_count} doc / ${payload.web_count} web).`;
    setStatus(message, "success");
    showEditorBanner("Sources refreshed successfully.", "success");
  }

  sectionsEl.addEventListener("click", async (event) => {
    const toggleButton = event.target.closest(".change-toggle");
    if (toggleButton) {
      const toggleSectionId = toggleButton.getAttribute("data-section-id");
      if (!toggleSectionId || !sectionDiffState[toggleSectionId]) return;
      sectionDiffState[toggleSectionId].showChanges = !sectionDiffState[toggleSectionId].showChanges;
      render();
      return;
    }

    const button = event.target.closest(".regen-btn");
    if (!button) {
      const historyButton = event.target.closest(".history-revert-btn");
      if (!historyButton) return;
      const versionId = historyButton.getAttribute("data-version-id");
      if (!versionId) return;
      try {
        await revertToVersion(versionId, historyButton);
      } catch (error) {
        const friendly = friendlyEditorMessage("revert", error.message || "");
        setStatus(friendly, "error");
        showEditorBanner(friendly, "error");
      }
      return;
    }

    const sectionId = button.getAttribute("data-section-id");
    if (!sectionId) return;

    const card = button.closest(".section-card");
    const instructionsInput = card ? card.querySelector(".section-guidance") : null;
    const instructions = instructionsInput ? String(instructionsInput.value || "").trim() : "";

    setButtonBusy(button, true, "Improving...");
    if (card) {
      card.classList.add("section-card-loading");
    }
    try {
      await regenerateSection(sectionId, instructions);
    } catch (error) {
      const friendly = friendlyEditorMessage("section", error.message || "");
      setStatus(friendly, "error");
      showEditorBanner(friendly, "error");
    }
    setButtonBusy(button, false);
    if (card) {
      card.classList.remove("section-card-loading");
    }
  });

  sectionsEl.addEventListener("input", function (event) {
    const input = event.target.closest(".section-guidance");
    if (!input) return;
    const sectionId = input.getAttribute("data-section-id");
    if (!sectionId) return;
    guidanceDrafts[sectionId] = String(input.value || "");
  });

  refreshBtn.addEventListener("click", async () => {
    try {
      await refreshSources();
    } catch (error) {
      const friendly = friendlyEditorMessage("refresh", error.message || "");
      setStatus(friendly, "error");
      showEditorBanner(friendly, "error");
    }
  });

  if (scrollTopEl) {
    scrollTopEl.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  if (shareBtn) {
    shareBtn.addEventListener("click", async function () {
      try {
        await toggleShare(shareBtn);
      } catch (error) {
        const friendly = friendlyEditorMessage("share", error.message || "");
        setStatus(friendly, "error");
        showEditorBanner(friendly, "error");
      }
    });
  }

  if (copyShareBtn) {
    copyShareBtn.addEventListener("click", async function () {
      const shareUrl = currentShareUrl();
      if (!shareUrl) {
        setStatus("Turn sharing on first to copy the public link.", "error");
        showEditorBanner("Turn sharing on first to copy the public link.", "error");
        return;
      }
      try {
        await copyText(shareUrl);
        setStatus("Shared link copied.", "success");
        showEditorBanner("Shared link copied.", "success");
      } catch (_error) {
        setStatus("Could not copy the shared link automatically.", "error");
        showEditorBanner("Could not copy the shared link automatically.", "error");
      }
    });
  }

  if (assignmentTopBtn) {
    assignmentTopBtn.addEventListener("click", async function () {
      setButtonBusy(assignmentTopBtn, true, "Creating...");
      try {
        await generateAssignment();
      } catch (error) {
        const friendly = friendlyEditorMessage("assignment", error.message || "");
        setAssignmentStatus(friendly, "error");
        showEditorBanner(friendly, "error");
      }
      setButtonBusy(assignmentTopBtn, false);
    });
  }

  if (assignmentGenerateBtn) {
    assignmentGenerateBtn.addEventListener("click", async function () {
      setButtonBusy(assignmentGenerateBtn, true, "Creating...");
      try {
        await generateAssignment();
      } catch (error) {
        const friendly = friendlyEditorMessage("assignment", error.message || "");
        setAssignmentStatus(friendly, "error");
        showEditorBanner(friendly, "error");
      }
      setButtonBusy(assignmentGenerateBtn, false);
    });
  }

  if (assignmentGradeBtn) {
    assignmentGradeBtn.addEventListener("click", async function () {
      setButtonBusy(assignmentGradeBtn, true, "Grading...");
      try {
        await gradeAssignment();
      } catch (error) {
        const friendly = friendlyEditorMessage("grading", error.message || "");
        setAssignmentStatus(friendly, "error");
        showEditorBanner(friendly, "error");
      }
      setButtonBusy(assignmentGradeBtn, false);
    });
  }

  if (assignmentExportMarkdownBtn) {
    assignmentExportMarkdownBtn.addEventListener("click", async function () {
      setButtonBusy(assignmentExportMarkdownBtn, true, "Exporting...");
      try {
        await exportAssignmentMarkdown();
      } catch (error) {
        const friendly = friendlyEditorMessage("export", error.message || "");
        setAssignmentStatus(friendly, "error");
        showEditorBanner(friendly, "error");
      }
      setButtonBusy(assignmentExportMarkdownBtn, false);
    });
  }

  if (assignmentExportJsonBtn) {
    assignmentExportJsonBtn.addEventListener("click", async function () {
      setButtonBusy(assignmentExportJsonBtn, true, "Exporting...");
      try {
        await exportAssignmentJson();
      } catch (error) {
        const friendly = friendlyEditorMessage("export", error.message || "");
        setAssignmentStatus(friendly, "error");
        showEditorBanner(friendly, "error");
      }
      setButtonBusy(assignmentExportJsonBtn, false);
    });
  }

  if (tutorModesEl) {
    tutorModesEl.addEventListener("change", function () {
      updateTutorModeUI();
    });
  }

  if (tutorSuggestionsEl) {
    tutorSuggestionsEl.addEventListener("click", async function (event) {
      const button = event.target.closest(".suggestion-btn");
      if (!button) return;
      const question = String(button.getAttribute("data-question") || "").trim();
      if (!question) return;
      setTutorMode("default");
      if (tutorQuestionEl) {
        tutorQuestionEl.value = question;
      }
      if (tutorAskBtn) {
        setButtonBusy(tutorAskBtn, true, "Thinking...");
      }
      try {
        await submitTutorRequest({ question: question, mode: "default" });
      } catch (error) {
        const friendly = friendlyEditorMessage("tutor", error.message || "");
        setStatus(friendly, "error");
        showEditorBanner(friendly, "error");
      }
      setButtonBusy(tutorAskBtn, false);
    });
  }

  if (tutorAskBtn) {
    tutorAskBtn.addEventListener("click", async function () {
      setButtonBusy(tutorAskBtn, true, "Thinking...");
      try {
        await submitTutorRequest({});
      } catch (error) {
        const friendly = friendlyEditorMessage("tutor", error.message || "");
        setStatus(friendly, "error");
        showEditorBanner(friendly, "error");
      }
      setButtonBusy(tutorAskBtn, false);
    });
  }

  const params = new URLSearchParams(window.location.search);
  if (params.get("created") === "1") {
    showEditorBanner("Module created successfully.", "success");
  }

  render();
})();
