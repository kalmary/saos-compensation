const slides = [...document.querySelectorAll('.slide')];
const stage = document.querySelector('#stage');
const previous = document.querySelector('#previous');
const next = document.querySelector('#next');
const counter = document.querySelector('#counter');
const announcement = document.querySelector('#announcement');
const indexDialog = document.querySelector('#index-dialog');
const notesDialog = document.querySelector('#notes-dialog');
const indexList = document.querySelector('#index-list');
const fullscreen = document.querySelector('#fullscreen');
let current = 0;
let opener = null;

/** Scale the slide canvas to the available window.
 * @param {void} none No arguments.
 * @returns {void} No return value.
 */
function resize() {
  const scale = Math.min(window.innerWidth / 1600, (window.innerHeight - 58) / 900);
  stage.style.transform = `scale(${scale})`;
  stage.style.marginLeft = `${(window.innerWidth - 1600 * scale) / 2}px`;
  stage.style.marginTop = `${Math.max(0, (window.innerHeight - 58 - 900 * scale) / 2)}px`;
}

/** Show a slide and synchronize navigation and accessibility state.
 * @param {number} index Zero-based slide index.
 * @param {boolean} updateHash Whether to update the URL hash.
 * @returns {void} No return value.
 */
function showSlide(index, updateHash = true) {
  current = Math.max(0, Math.min(slides.length - 1, index));
  slides.forEach((slide, position) => { slide.hidden = position !== current; });
  previous.disabled = current === 0;
  next.disabled = current === slides.length - 1;
  counter.textContent = `${String(current + 1).padStart(2, '0')} / ${slides.length}`;
  const title = slides[current].querySelector('h1, h2').textContent.trim();
  document.title = `${title} — SAOS Compensation`;
  announcement.textContent = `Slajd ${current + 1} z ${slides.length}: ${title}`;
  indexList.querySelectorAll('button').forEach((button, position) => {
    if (position === current) button.setAttribute('aria-current', 'true');
    else button.removeAttribute('aria-current');
  });
  if (updateHash && location.hash !== `#slide-${current + 1}`) history.replaceState(null, '', `#slide-${current + 1}`);
}

/** Read and display the slide identified by the URL hash.
 * @param {void} none No arguments.
 * @returns {void} No return value.
 */
function readHash() {
  const match = /^#slide-(\d+)$/.exec(location.hash);
  showSlide(match ? Number(match[1]) - 1 : 0, false);
}

/** Open a dialog and remember the initiating control.
 * @param {HTMLDialogElement} dialog Dialog to open.
 * @param {HTMLElement} button Initiating control.
 * @returns {void} No return value.
 */
function openDialog(dialog, button) {
  opener = button;
  dialog.showModal();
  dialog.querySelector('.close').focus();
}

slides.forEach((slide, index) => {
  const number = document.createElement('span');
  number.className = 'slide-number';
  number.textContent = `${String(index + 1).padStart(2, '0')} / ${slides.length}`;
  slide.append(number);
  const item = document.createElement('li');
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = slide.querySelector('h1, h2').textContent.trim();
  button.addEventListener('click', () => { indexDialog.close(); showSlide(index); });
  item.append(button);
  indexList.append(item);
});
previous.addEventListener('click', () => showSlide(current - 1));
next.addEventListener('click', () => showSlide(current + 1));
document.querySelector('#index-button').addEventListener('click', (event) => openDialog(indexDialog, event.currentTarget));
document.querySelector('#notes-button').addEventListener('click', (event) => {
  document.querySelector('#notes-slide-title').textContent = slides[current].querySelector('h1, h2').textContent.trim();
  document.querySelector('#notes-text').textContent = slides[current].dataset.notes;
  document.querySelector('#notes-sources').textContent = slides[current].dataset.sources;
  openDialog(notesDialog, event.currentTarget);
});
document.querySelectorAll('dialog').forEach((dialog) => {
  dialog.querySelector('.close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => { if (opener) opener.focus(); });
});
if (!document.fullscreenEnabled) fullscreen.hidden = true;
fullscreen.addEventListener('click', () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen();
});
document.addEventListener('fullscreenchange', () => { fullscreen.textContent = document.fullscreenElement ? 'Opuść pełny ekran' : 'Pełny ekran'; });
document.addEventListener('keydown', (event) => {
  if (event.target.closest('input, textarea, select, [contenteditable="true"]') || document.querySelector('dialog[open]')) return;
  if (event.key === ' ' && event.target.closest('button')) return;
  const keys = { ArrowRight: current + 1, ' ': current + 1, PageDown: current + 1, ArrowLeft: current - 1, PageUp: current - 1, Home: 0, End: slides.length - 1 };
  if (Object.hasOwn(keys, event.key)) { event.preventDefault(); showSlide(keys[event.key]); }
});
window.addEventListener('hashchange', readHash);
window.addEventListener('resize', resize);
resize();
readHash();
