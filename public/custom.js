/**
 * Lecture Intelligence System - Custom Audio Seek Script
 * 
 * Enables clickable timestamps in AI responses to seek to
 * specific parts of uploaded lecture recordings.
 */

/**
 * Seek all audio/video elements on the page to a given timestamp.
 * @param {number} seconds - The timestamp in seconds to seek to.
 */
function seekAudioTo(seconds) {
    const mediaElements = document.querySelectorAll('audio, video');

    if (mediaElements.length === 0) {
        console.warn('[LIS] No audio/video elements found on the page.');
        return;
    }

    mediaElements.forEach((media) => {
        if (media.readyState >= 1) {
            media.currentTime = seconds;
            media.play();
        } else {
            media.addEventListener('loadedmetadata', () => {
                media.currentTime = seconds;
                media.play();
            }, { once: true });
        }
    });
}

window.seekAudioTo = seekAudioTo;

/**
 * MutationObserver: Watches for new timestamp elements added to the chat
 * and attaches click handlers to them automatically.
 * This avoids relying on inline onclick which Chainlit strips.
 */
function attachTimestampListeners() {
    const timestamps = document.querySelectorAll('.seek-timestamp:not([data-bound])');
    timestamps.forEach((el) => {
        el.setAttribute('data-bound', 'true');
        el.addEventListener('click', (e) => {
            e.preventDefault();
            const seconds = parseFloat(el.getAttribute('data-seconds'));
            if (!isNaN(seconds)) {
                seekAudioTo(seconds);
            }
        });
    });
}

// Run on page load
attachTimestampListeners();

// Watch for new messages being added to the chat
const observer = new MutationObserver(() => {
    attachTimestampListeners();
});

// Start observing once the DOM is ready
function startObserving() {
    const target = document.body;
    if (target) {
        observer.observe(target, { childList: true, subtree: true });
    } else {
        setTimeout(startObserving, 500);
    }
}

startObserving();
