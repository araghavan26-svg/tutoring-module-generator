(function () {
  const STORAGE_KEY = "moduleforge-theme";

  function normalizeTheme(theme) {
    return theme === "dark" ? "dark" : "light";
  }

  function getStoredTheme() {
    try {
      return normalizeTheme(localStorage.getItem(STORAGE_KEY));
    } catch (_) {
      return "light";
    }
  }

  function saveTheme(theme) {
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch (_) {
      // Local storage can be unavailable in private or restricted browser modes.
    }
  }

  function updateToggle(theme) {
    const toggle = document.getElementById("theme-toggle");
    const label = document.getElementById("theme-toggle-label");
    if (!toggle || !label) return;

    const isDark = theme === "dark";
    toggle.setAttribute("aria-pressed", String(isDark));
    toggle.setAttribute("aria-label", isDark ? "Switch to light mode" : "Switch to dark mode");
    label.textContent = isDark ? "Light mode" : "Dark mode";
  }

  function applyTheme(theme, shouldSave) {
    const normalized = normalizeTheme(theme);
    document.documentElement.dataset.theme = normalized;
    if (shouldSave) {
      saveTheme(normalized);
    }
    updateToggle(normalized);
  }

  document.addEventListener("DOMContentLoaded", function () {
    const initialTheme = getStoredTheme();
    applyTheme(initialTheme, false);

    const toggle = document.getElementById("theme-toggle");
    if (!toggle) return;

    toggle.addEventListener("click", function () {
      const currentTheme = normalizeTheme(document.documentElement.dataset.theme);
      applyTheme(currentTheme === "dark" ? "light" : "dark", true);
    });
  });
})();
