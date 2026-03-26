(function () {
  const form = document.getElementById("create-form");
  if (!form) return;

  const statusEl = document.getElementById("create-status");
  const errorEl = document.getElementById("create-error");
  const demoBannerEl = document.getElementById("create-demo-banner");
  const sampleFlowNoteEl = document.getElementById("sample-flow-note");
  const progressBoxEl = document.getElementById("create-progress");
  const progressMessageEl = document.getElementById("create-progress-message");
  const progressBarEl = document.getElementById("create-progress-bar");
  const submitBtn = document.getElementById("create-submit");
  const progressEl = document.getElementById("wizard-progress");

  const learningRequestEl = document.getElementById("learning_request");
  const learnerLevelEl = document.getElementById("audience_level");
  const customLevelWrapEl = document.getElementById("custom_level_wrap");
  const customLevelDescriptionEl = document.getElementById("custom_level_description");
  const familiarityEl = document.getElementById("current_familiarity");
  const goalsEl = document.getElementById("learning_objectives");
  const learningPurposeEl = document.getElementById("learning_purpose");
  const explanationStyleEl = document.getElementById("explanation_style");
  const confusionPointsEl = document.getElementById("confusion_points");
  const allowWebEl = document.getElementById("allow_web");
  const sourcePreferenceEl = document.getElementById("source_preference");
  const preferHighTrustEl = document.getElementById("prefer_high_trust_sources");
  const allowedDomainsEl = document.getElementById("allowed_domains");
  const blockedDomainsEl = document.getElementById("blocked_domains");
  const filesEl = document.getElementById("files");
  const useSampleDocEl = document.getElementById("use_sample_doc");
  const relatedToPreviousEl = document.getElementById("related_to_previous");
  const presetLearnScratchEl = document.getElementById("preset-learn-scratch");
  const presetStudyTestEl = document.getElementById("preset-study-test");
  const presetQuickReviewEl = document.getElementById("preset-quick-review");

  const SAMPLE_LEARNING_REQUEST = "Biology - Photosynthesis";
  const SAMPLE_GOALS = [
    "Explain what photosynthesis is",
    "Identify key vocabulary",
    "Apply the idea in a simple example",
  ];
  const PROGRESS_STAGES = [
    "Finding sources...",
    "Reading your materials...",
    "Building lesson sections...",
    "Generating practice questions...",
  ];
  const PROGRESS_WIDTHS = ["18%", "42%", "68%", "90%"];
  const MODULE_TIMEOUT_MESSAGE = "Module generation took too long. Please try fewer goals or fewer sources.";
  let progressTimer = null;
  let progressStageIndex = 0;

  function setStatus(message, kind) {
    statusEl.textContent = message || "";
    statusEl.className = "status" + (kind ? " " + kind : "");
  }

  function showDemoBanner(message) {
    if (!demoBannerEl) return;
    demoBannerEl.textContent = message || "";
    demoBannerEl.hidden = !message;
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

  function clearProgressTimer() {
    if (!progressTimer) return;
    window.clearInterval(progressTimer);
    progressTimer = null;
  }

  function setProgressStage(index) {
    progressStageIndex = Math.max(0, Math.min(PROGRESS_STAGES.length - 1, Number(index) || 0));
    if (progressMessageEl) {
      progressMessageEl.textContent = PROGRESS_STAGES[progressStageIndex];
    }
    if (progressBarEl) {
      progressBarEl.style.width = PROGRESS_WIDTHS[progressStageIndex];
    }
  }

  function setLoading(isLoading, stageIndex) {
    if (!progressBoxEl) return;
    if (!isLoading) {
      clearProgressTimer();
      progressBoxEl.hidden = true;
      progressBoxEl.classList.remove("is-running");
      if (progressBarEl) {
        progressBarEl.style.width = "0%";
      }
      return;
    }

    progressBoxEl.hidden = false;
    progressBoxEl.classList.add("is-running");
    setProgressStage(stageIndex || 0);
    clearProgressTimer();
    progressTimer = window.setInterval(function () {
      const nextIndex = (progressStageIndex + 1) % PROGRESS_STAGES.length;
      setProgressStage(nextIndex);
    }, 1700);
  }

  function setLoadingStage(index) {
    if (!progressBoxEl || progressBoxEl.hidden) return;
    setProgressStage(index);
  }

  function showError(message) {
    errorEl.textContent = message;
    errorEl.hidden = false;
  }

  function clearError() {
    errorEl.hidden = true;
    errorEl.textContent = "";
  }

  function splitLines(text) {
    return String(text || "")
      .split(/\n|,/)
      .map((item) => item.trim())
      .filter(Boolean);
  }

  function shouldUseSampleMode() {
    const params = new URLSearchParams(window.location.search);
    return params.get("sample") === "1" || params.get("demo") === "1";
  }

  function applySampleDefaults() {
    if (!shouldUseSampleMode()) return;

    if (learningRequestEl) {
      learningRequestEl.value = SAMPLE_LEARNING_REQUEST;
    }
    learnerLevelEl.value = "Beginner";
    if (customLevelDescriptionEl) {
      customLevelDescriptionEl.value = "";
    }
    if (familiarityEl) {
      familiarityEl.value = "Brand new";
    }
    goalsEl.value = SAMPLE_GOALS.join("\n");
    if (learningPurposeEl) {
      learningPurposeEl.value = "Learn from scratch";
    }
    if (explanationStyleEl) {
      explanationStyleEl.value = "Step-by-step";
    }
    if (confusionPointsEl) {
      confusionPointsEl.value = "I want extra help connecting sunlight, chlorophyll, and glucose production.";
    }
    allowWebEl.checked = true;
    if (sourcePreferenceEl) {
      sourcePreferenceEl.value = "Beginner-friendly";
    }
    if (preferHighTrustEl) {
      preferHighTrustEl.checked = true;
    }
    if (allowedDomainsEl) {
      allowedDomainsEl.value = "kids.britannica.com\nwww.vocabulary.com";
    }
    if (blockedDomainsEl) {
      blockedDomainsEl.value = "";
    }
    if (useSampleDocEl && window.SAMPLE_DOC_EXISTS) {
      useSampleDocEl.checked = true;
    }
    if (sampleFlowNoteEl) {
      sampleFlowNoteEl.hidden = false;
    }
    showDemoBanner("Demo mode is ready. A polished sample topic is prefilled, and the sample document can be added automatically.");
    if (submitBtn) {
      submitBtn.textContent = "Generate Sample Module";
    }
    setActivePreset("preset-learn-scratch");
    toggleCustomLevelField();
  }

  function setActivePreset(activeId) {
    [presetLearnScratchEl, presetStudyTestEl, presetQuickReviewEl].forEach((button) => {
      if (!button) return;
      button.classList.toggle("is-active", button.id === activeId);
    });
  }

  function applyPreset(preset) {
    if (!familiarityEl || !learningPurposeEl || !explanationStyleEl) return;

    if (preset === "scratch") {
      learnerLevelEl.value = "Beginner";
      familiarityEl.value = "Brand new";
      learningPurposeEl.value = "Learn from scratch";
      explanationStyleEl.value = "Simple and beginner-friendly";
      setActivePreset("preset-learn-scratch");
      setStatus("Preset applied: Learn from scratch.", "info");
    } else if (preset === "test") {
      if (learnerLevelEl.value === "Beginner") {
        learnerLevelEl.value = "Intermediate";
      }
      familiarityEl.value = "I know the basics";
      learningPurposeEl.value = "Review for a test";
      explanationStyleEl.value = "Concise review";
      setActivePreset("preset-study-test");
      setStatus("Preset applied: Study for a test.", "info");
    } else if (preset === "review") {
      familiarityEl.value = "I know the basics";
      learningPurposeEl.value = "Practice applying ideas";
      explanationStyleEl.value = "Concise review";
      setActivePreset("preset-quick-review");
      setStatus("Preset applied: Quick review.", "info");
    }

    toggleCustomLevelField();
    refreshProgressFromInputs();
  }

  function toggleCustomLevelField() {
    const isCustom = learnerLevelEl && learnerLevelEl.value === "Custom";
    if (customLevelWrapEl) {
      customLevelWrapEl.hidden = !isCustom;
    }
    if (customLevelDescriptionEl) {
      customLevelDescriptionEl.required = Boolean(isCustom);
      if (!isCustom) {
        customLevelDescriptionEl.value = "";
      }
    }
  }

  function resolvedAudienceLevel() {
    const selectedLevel = learnerLevelEl ? String(learnerLevelEl.value || "").trim() : "";
    if (selectedLevel === "Custom") {
      return customLevelDescriptionEl ? String(customLevelDescriptionEl.value || "").trim() : "";
    }
    return selectedLevel || "General";
  }

  function setProgress(step) {
    if (!progressEl) return;
    const current = Math.max(1, Math.min(4, Number(step) || 1));
    const items = Array.from(progressEl.querySelectorAll("li[data-step]"));
    items.forEach((item) => {
      const itemStep = Number(item.getAttribute("data-step") || "0");
      item.classList.toggle("done", itemStep < current);
      item.classList.toggle("active", itemStep === current);
    });
  }

  function refreshProgressFromInputs() {
    const learningRequestReady = learningRequestEl.value.trim().length > 0;
    const goalsReady = splitLines(goalsEl.value).length > 0;
    if (!learningRequestReady) {
      setProgress(1);
      return;
    }
    if (!goalsReady) {
      setProgress(2);
      return;
    }
    setProgress(3);
  }

  function friendlyErrorMessage(context, detail) {
    const text = String(detail || "").toLowerCase();
    if (text.includes("took too long") || text.includes("timed out") || text.includes("504")) {
      return MODULE_TIMEOUT_MESSAGE;
    }
    if (text.includes("missing openai_api_key")) {
      return "The app is missing an OpenAI API key. Add your key to the .env file and restart.";
    }
    if (text.includes("evidence retrieval failed")) {
      return "We could not gather sources right now. Try again, or adjust source settings.";
    }
    if (text.includes("module generation failed")) {
      return "We could not generate the module right now. Please try again in a moment.";
    }
    if (context === "upload") {
      return "We could not upload your document(s). Please check file type and try again.";
    }
    return "Something went wrong. Please try again.";
  }

  async function parseError(response) {
    try {
      const payload = await response.json();
      if (payload && typeof payload.detail !== "undefined") {
        return String(payload.detail);
      }
    } catch (_err) {
      // ignore
    }
    return response.statusText || "Request failed";
  }

  async function appendSampleDocument(formData) {
    if (!useSampleDocEl || !useSampleDocEl.checked || !window.SAMPLE_DOC_EXISTS) {
      return;
    }
    const sampleResp = await fetch("/samples/sample.txt");
    if (!sampleResp.ok) {
      throw new Error("sample-download-failed");
    }
    const sampleBlob = await sampleResp.blob();
    const sampleFile = new File([sampleBlob], "sample.txt", { type: "text/plain" });
    formData.append("files", sampleFile);
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearError();
    setStatus("", "");
    setLoading(false);
    setButtonBusy(submitBtn, true, "Generating...");
    setActivePreset("");

    setProgress(4);
    const learningRequest = learningRequestEl ? learningRequestEl.value.trim() : "";
    const learnerLevel = learnerLevelEl ? learnerLevelEl.value.trim() : "";
    const customLevelDescription = customLevelDescriptionEl ? customLevelDescriptionEl.value.trim() : "";
    const audienceLevel = resolvedAudienceLevel();
    const currentFamiliarity = familiarityEl ? familiarityEl.value.trim() : "";
    const objectives = splitLines(goalsEl.value);
    const learningPurpose = learningPurposeEl ? learningPurposeEl.value.trim() : "";
    const explanationStyle = explanationStyleEl ? explanationStyleEl.value.trim() : "";
    const confusionPoints = confusionPointsEl ? confusionPointsEl.value.trim() : "";
    const allowWeb = !!allowWebEl.checked;
    const sourcePreference = sourcePreferenceEl ? sourcePreferenceEl.value.trim() : "General";
    const preferHighTrustSources = !!(preferHighTrustEl && preferHighTrustEl.checked);
    const allowedDomainsRaw = allowedDomainsEl ? splitLines(allowedDomainsEl.value) : [];
    const blockedDomainsRaw = blockedDomainsEl ? splitLines(blockedDomainsEl.value) : [];

    if (!learningRequest) {
      showError("Please enter what you want to learn.");
      setButtonBusy(submitBtn, false);
      setProgress(1);
      return;
    }
    if (objectives.length === 0) {
      showError("Please add at least one learning goal.");
      setButtonBusy(submitBtn, false);
      setProgress(2);
      return;
    }
    if (learnerLevel === "Custom" && !customLevelDescription) {
      showError("Please describe your level.");
      setButtonBusy(submitBtn, false);
      setProgress(1);
      return;
    }

    try {
      let vectorStoreId = null;
      const formData = new FormData();

      const selectedFiles = filesEl && filesEl.files ? Array.from(filesEl.files) : [];
      selectedFiles.forEach((file) => formData.append("files", file));
      await appendSampleDocument(formData);

      setLoading(true, 0);
      setStatus(shouldUseSampleMode() ? "Preparing the sample module..." : "Preparing your module...", "loading");

      if (Array.from(formData.keys()).length > 0) {
        setLoadingStage(1);
        setStatus("Reading your materials...", "loading");
        const uploadResp = await fetch("/v1/docs/upload", {
          method: "POST",
          body: formData,
        });
        if (!uploadResp.ok) {
          const detail = await parseError(uploadResp);
          throw new Error("upload:" + detail);
        }
        const uploadData = await uploadResp.json();
        vectorStoreId = uploadData.vector_store_id || null;
      }

      setLoadingStage(2);
      setStatus("Building lesson sections from grounded evidence...", "loading");
      const sourcePolicy = {
        allow_web: allowWeb,
        web_recency_days: 30,
        allowed_domains: allowedDomainsRaw.length > 0 ? allowedDomainsRaw : null,
        blocked_domains: blockedDomainsRaw.length > 0 ? blockedDomainsRaw : null,
      };

      const generateResp = await fetch("/v1/modules/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          learning_request: learningRequest,
          audience_level: audienceLevel,
          learner_level: learnerLevel || null,
          custom_level_description: customLevelDescription || null,
          current_familiarity: currentFamiliarity || null,
          learning_purpose: learningPurpose || null,
          explanation_style: explanationStyle || null,
          confusion_points: confusionPoints || null,
          source_preference: sourcePreference || "General",
          prefer_high_trust_sources: preferHighTrustSources,
          learning_objectives: objectives,
          allow_web: allowWeb,
          vector_store_id: vectorStoreId,
          source_policy: sourcePolicy,
          related_to_previous: relatedToPreviousEl ? relatedToPreviousEl.checked : false,
          fast_mode: true,
        }),
      });

      if (!generateResp.ok) {
        const detail = await parseError(generateResp);
        throw new Error("generate:" + detail);
      }

      const generateData = await generateResp.json();
      const moduleId = generateData?.module?.module_id;
      if (!moduleId) {
        throw new Error("generate:missing-module-id");
      }

      setLoadingStage(3);
      setStatus("Module created successfully. Opening editor...", "success");
      setLoading(false);
      localStorage.setItem("module_" + moduleId, JSON.stringify(generateData));
      window.location.assign("/modules/" + moduleId + "?created=1");
      return;
    } catch (error) {
      const rawMessage = String(error && error.message ? error.message : "");
      if (rawMessage.startsWith("upload:")) {
        showError(friendlyErrorMessage("upload", rawMessage.slice(7)));
      } else if (rawMessage.startsWith("generate:")) {
        showError(friendlyErrorMessage("generate", rawMessage.slice(9)));
      } else {
        showError(friendlyErrorMessage("general", rawMessage));
      }
      setStatus("", "");
      setLoading(false);
      setButtonBusy(submitBtn, false);
      refreshProgressFromInputs();
      return;
    }
  });

  [learningRequestEl, learnerLevelEl, customLevelDescriptionEl, familiarityEl, goalsEl, learningPurposeEl, explanationStyleEl, confusionPointsEl, allowWebEl, sourcePreferenceEl, preferHighTrustEl, allowedDomainsEl, blockedDomainsEl].forEach((el) => {
    if (!el) return;
    el.addEventListener("input", refreshProgressFromInputs);
    el.addEventListener("change", refreshProgressFromInputs);
  });

  if (learnerLevelEl) {
    learnerLevelEl.addEventListener("change", function () {
      toggleCustomLevelField();
      setActivePreset("");
    });
  }
  [familiarityEl, learningPurposeEl, explanationStyleEl].forEach((el) => {
    if (!el) return;
    el.addEventListener("change", function () {
      setActivePreset("");
    });
  });
  if (presetLearnScratchEl) {
    presetLearnScratchEl.addEventListener("click", function () {
      applyPreset("scratch");
    });
  }
  if (presetStudyTestEl) {
    presetStudyTestEl.addEventListener("click", function () {
      applyPreset("test");
    });
  }
  if (presetQuickReviewEl) {
    presetQuickReviewEl.addEventListener("click", function () {
      applyPreset("review");
    });
  }

  toggleCustomLevelField();
  applySampleDefaults();
  refreshProgressFromInputs();
})();
