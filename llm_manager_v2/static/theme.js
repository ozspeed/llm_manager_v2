// Theme toggle logic for LLM Model Manager v2
// [V2-7.1.7] Modern UI/UX
(function() {
  const themeToggle = document.getElementById('theme-toggle');
  const root = document.documentElement;
  // Detect system preference on first load
  function getPreferredTheme() {
    if (window.localStorage.getItem('theme')) {
      return window.localStorage.getItem('theme');
    }
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  function setTheme(theme) {
    root.setAttribute('data-theme', theme);
    window.localStorage.setItem('theme', theme);
    if (themeToggle) {
      themeToggle.setAttribute('aria-label', theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
      themeToggle.innerHTML = theme === 'dark' ? '☀️' : '🌙';
    }
  }
  if (themeToggle) {
    themeToggle.addEventListener('click', function() {
      const newTheme = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      setTheme(newTheme);
    });
  }
  setTheme(getPreferredTheme());
})();
