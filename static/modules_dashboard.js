(function () {
  const statusEl = document.getElementById("library-status");
  const libraryEl = document.getElementById("module-library");
  const emptyStateEl = document.getElementById("empty-library");
  const bannerEl = document.getElementById("library-banner");
  if (!statusEl) return;

  function setStatus(message, kind) {
    statusEl.textContent = message || "";
    statusEl.className = "status" + (kind ? " " + kind : "");
  }

  function showBanner(message, kind) {
    if (!bannerEl) return;
    bannerEl.textContent = message || "";
    bannerEl.className = "alert" + (kind ? " " + kind : " info");
    bannerEl.hidden = !message;
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

  function friendlyLibraryMessage(context, detail) {
    const text = String(detail || "").toLowerCase();
    if (text.includes("took too long") || text.includes("timed out") || text.includes("504")) {
      return "That action took too long. Please try again.";
    }
    if (context === "share") {
      return "We couldn't update sharing right now. Please try again.";
    }
    if (context === "delete") {
      return "We couldn't delete that module right now. Please try again.";
    }
    return "Something went wrong. Please try again.";
  }

  function syncEmptyState() {
    const hasCards = Boolean(libraryEl && libraryEl.querySelector(".library-card"));
    if (libraryEl) {
      libraryEl.hidden = !hasCards;
    }
    if (emptyStateEl) {
      emptyStateEl.hidden = hasCards;
    }
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

  async function parseError(response) {
    try {
      const payload = await response.json();
      return payload && payload.detail ? String(payload.detail) : "Request failed";
    } catch (_err) {
      return response.statusText || "Request failed";
    }
  }

  async function deleteModule(moduleId) {
    const response = await fetch(`/v1/modules/${moduleId}/delete`, {
      method: "POST",
    });
    if (!response.ok) {
      throw new Error(await parseError(response));
    }
    return response.json();
  }

  async function toggleShare(moduleId, enabled) {
    const response = await fetch(`/v1/modules/${moduleId}/share`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: enabled }),
    });
    if (!response.ok) {
      throw new Error(await parseError(response));
    }
    return response.json();
  }

  function updateShareCard(card, payload) {
    if (!card) return;
    const shareEnabled = Boolean(payload.share_enabled);
    const shareUrl = String(payload.share_url || "");
    const shareBadge = card.querySelector(".library-share-badge");
    const shareMeta = card.querySelector(".share-state");
    const shareWrap = card.querySelector(".library-share");
    const shareInput = card.querySelector(".share-link-input");
    const openLink = card.querySelector(".open-share-link");
    const shareButton = card.querySelector(".share-module-btn");
    const copyButton = card.querySelector(".copy-share-btn");

    if (shareBadge) {
      shareBadge.hidden = !shareEnabled;
    }
    if (shareMeta) {
      shareMeta.textContent = `Sharing: ${shareEnabled ? "On" : "Off"}`;
    }
    if (shareWrap) {
      shareWrap.hidden = !shareEnabled;
    }
    if (shareInput) {
      shareInput.value = shareEnabled ? shareUrl : "";
    }
    if (openLink) {
      openLink.href = shareEnabled ? shareUrl : "#";
    }
    if (shareButton) {
      shareButton.textContent = shareEnabled ? "Disable sharing" : "Share module";
      shareButton.setAttribute("data-share-enabled", shareEnabled ? "true" : "false");
    }
    if (copyButton) {
      copyButton.hidden = !shareEnabled;
    }
  }

  document.addEventListener("click", async function (event) {
    const copyButton = event.target.closest(".copy-share-btn");
    if (copyButton) {
      const card = copyButton.closest(".library-card");
      const input = card ? card.querySelector(".share-link-input") : null;
      const shareUrl = input ? String(input.value || "").trim() : "";
      if (!shareUrl) {
        setStatus("Turn sharing on first to copy the public link.", "error");
        return;
      }
      try {
        await copyText(shareUrl);
        setStatus("Shared link copied.", "success");
        showBanner("Shared link copied.", "success");
      } catch (_error) {
        setStatus("Could not copy the shared link automatically.", "error");
        showBanner("Could not copy the shared link automatically.", "error");
      }
      return;
    }

    const shareButton = event.target.closest(".share-module-btn");
    if (shareButton) {
      const moduleId = String(shareButton.getAttribute("data-module-id") || "").trim();
      if (!moduleId) return;
      const card = shareButton.closest(".library-card");
      const enabling = shareButton.getAttribute("data-share-enabled") !== "true";
      setButtonBusy(shareButton, true, enabling ? "Sharing..." : "Turning off...");
      setStatus("");
      showBanner(enabling ? "Turning sharing on..." : "Turning sharing off...", "info");
      setStatus(enabling ? "Turning sharing on..." : "Turning sharing off...", "loading");
      try {
        const payload = await toggleShare(moduleId, enabling);
        updateShareCard(card, payload);
        const message = payload.share_enabled
          ? "Sharing enabled."
          : "Sharing disabled. Existing shared links will no longer work.";
        setStatus(message, "success");
        showBanner(message, "success");
      } catch (error) {
        const friendly = friendlyLibraryMessage("share", error.message || "");
        setStatus(friendly, "error");
        showBanner(friendly, "error");
      }
      setButtonBusy(shareButton, false);
      return;
    }

    const button = event.target.closest(".delete-module-btn");
    if (!button) return;

    const moduleId = String(button.getAttribute("data-module-id") || "").trim();
    if (!moduleId) return;

    const confirmed = window.confirm("Delete this saved module and its history?");
    if (!confirmed) return;

    setButtonBusy(button, true, "Deleting...");
    setStatus("Deleting the saved module...", "loading");
    showBanner("Deleting the saved module...", "info");
    try {
      await deleteModule(moduleId);
      const card = button.closest(".library-card");
      if (card) {
        card.remove();
      }
      syncEmptyState();
      setStatus("Module deleted.", "success");
      showBanner("Module deleted.", "success");
    } catch (error) {
      const friendly = friendlyLibraryMessage("delete", error.message || "");
      setStatus(friendly, "error");
      showBanner(friendly, "error");
      setButtonBusy(button, false);
      return;
    }
    setButtonBusy(button, false);
  });

  syncEmptyState();
})();
