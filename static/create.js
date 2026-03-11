(function () {
  const form = document.getElementById("create-form");
  if (!form) return;

  const statusEl = document.getElementById("create-status");
  const errorEl = document.getElementById("create-error");
  const loadingEl = document.getElementById("create-loading");
  const submitBtn = document.getElementById("create-submit");
  const progressEl = document.getElementById("wizard-progress");

  const topicEl = document.getElementById("topic");
  const learnerLevelEl = document.getElementById("audience_level");
  const goalsEl = document.getElementById("learning_objectives");
  const allowWebEl = document.getElementById("allow_web");
  const allowedDomainsEl = document.getElementById("allowed_domains");
  const blockedDomainsEl = document.getElementById("blocked_domains");
  const filesEl = document.getElementById("files");
  const useSampleDocEl = document.getElementById("use_sample_doc");

  const SAMPLE_TOPIC = "Photosynthesis";
  const SAMPLE_GOALS = [
    "Explain what photosynthesis is",
    "Identify key vocabulary",
    "Apply the idea in a simple example",
  ];

  function setStatus(message, kind) {
    statusEl.textContent = message || "";
    statusEl.className = "status" + (kind ? " " + kind : "");
  }

  function setLoading(isLoading, message) {
    if (!loadingEl) return;
    loadingEl.hidden = !isLoading;
    if (isLoading && message) {
      loadingEl.querySelector("span:last-child").textContent = message;
    }
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

    topicEl.value = SAMPLE_TOPIC;
    learnerLevelEl.value = "Middle school";
    goalsEl.value = SAMPLE_GOALS.join("\n");
    allowWebEl.checked = true;
    if (allowedDomainsEl) {
      allowedDomainsEl.value = "kids.britannica.com\nwww.vocabulary.com";
    }
    if (blockedDomainsEl) {
      blockedDomainsEl.value = "";
    }
    if (useSampleDocEl && window.SAMPLE_DOC_EXISTS) {
      useSampleDocEl.checked = true;
    }
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
    const topicReady = topicEl.value.trim().length > 0;
    const goalsReady = splitLines(goalsEl.value).length > 0;
    if (!topicReady) {
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
    setLoading(false, "");
    submitBtn.disabled = true;

    setProgress(4);
    const topic = topicEl.value.trim();
    const audienceLevel = learnerLevelEl.value.trim() || "General";
    const objectives = splitLines(goalsEl.value);
    const allowWeb = !!allowWebEl.checked;
    const allowedDomainsRaw = allowedDomainsEl ? splitLines(allowedDomainsEl.value) : [];
    const blockedDomainsRaw = blockedDomainsEl ? splitLines(blockedDomainsEl.value) : [];

    if (!topic) {
      showError("Please enter a topic.");
      submitBtn.disabled = false;
      setProgress(1);
      return;
    }
    if (objectives.length === 0) {
      showError("Please add at least one learning goal.");
      submitBtn.disabled = false;
      setProgress(2);
      return;
    }

    try {
      let vectorStoreId = null;
      const formData = new FormData();

      const selectedFiles = filesEl && filesEl.files ? Array.from(filesEl.files) : [];
      selectedFiles.forEach((file) => formData.append("files", file));
      await appendSampleDocument(formData);

      if (Array.from(formData.keys()).length > 0) {
        setLoading(true, "Uploading sources...");
        setStatus("Uploading sources...", "");
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

      setLoading(true, "Building your module... this may take a moment.");
      setStatus("Generating your module...", "");
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
          topic,
          audience_level: audienceLevel,
          learning_objectives: objectives,
          allow_web: allowWeb,
          vector_store_id: vectorStoreId,
          source_policy: sourcePolicy,
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

      setStatus("Module created successfully. Opening editor...", "success");
      setLoading(false, "");
      localStorage.setItem("module_" + moduleId, JSON.stringify(generateData));
      window.location.assign("/modules/" + moduleId + "?created=1");
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
      setLoading(false, "");
      submitBtn.disabled = false;
      refreshProgressFromInputs();
      return;
    }

    submitBtn.disabled = false;
  });

  [topicEl, learnerLevelEl, goalsEl, allowWebEl, allowedDomainsEl, blockedDomainsEl].forEach((el) => {
    if (!el) return;
    el.addEventListener("input", refreshProgressFromInputs);
    el.addEventListener("change", refreshProgressFromInputs);
  });

  applySampleDefaults();
  refreshProgressFromInputs();
})();
