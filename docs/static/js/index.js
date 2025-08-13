function copyBibtex() {
	var copyText = document.getElementById("BibTeX");
	navigator.clipboard.writeText(copyText.innerHTML);
}

// Media controls for GIF animation
document.addEventListener('DOMContentLoaded', function() {
	let isPlaying = true;
	const DEMOS = {
		"constrained": "constrained-demo",
		"unconstrained": "unconstrained-demo"
	};
	let activeDemo = DEMOS["constrained"];
	let rubs = {};
	let activeRub = () => rubs[activeDemo];
	let isActiveRub = (rub) => activeRub() === rub;

	// Initialize the GIF player
	const imgTags = document.querySelectorAll('.figure-demo');
	imgTags.forEach((imgTag) => {
		if (/.*\.gif/.test(imgTag.src)) {
			let rub = new RubbableGif({
				gif: imgTag,
				loop_mode: false,
				auto_play: false,
				draw_while_loading: true,
				max_width: window.screen.width * 0.9,
				on_end: function() {
					console.log('GIF ended');
					pauseButton.click();
				}
			});
			rubs[imgTag.id] = rub;
			rub.load(function() {
				console.log('GIF loaded successfully');
				if (isActiveRub(rub)) {
					rub.get_canvas().style.display = 'unset';
					document.getElementById("loading-text").style.display = 'none';
					pauseButton.click();
					restartButton.click();
					playButton.click();
				} else {
					rub.pause();
					rub.get_canvas().style.display = 'none';
				}
			});
		}
	})

	// Get control buttons using querySelector by their icons
	const pauseButton = document.querySelector('a .fa-pause').parentElement;
	const playButton = document.querySelector('a .fa-play').parentElement;
	const restartButton = document.querySelector('a .fa-step-backward').parentElement;
	const previousButton = document.querySelector('a .fa-undo').parentElement;
	const nextButton = document.querySelector('a .fa-redo').parentElement;

	// Function to update button states
	function updateButtonStates() {
		if (pauseButton && playButton) {
			if (isPlaying) {
				pauseButton.parentElement.classList.remove('is-active');
				playButton.parentElement.classList.add('is-active');
			} else {
				pauseButton.parentElement.classList.add('is-active');
				playButton.parentElement.classList.remove('is-active');
			}
		}
	}

	// Add event listeners for media controls
	if (pauseButton) {
		pauseButton.addEventListener('click', function(e) {
			e.preventDefault();
			if (activeRub()) {
				activeRub().pause();
				isPlaying = false;
				updateButtonStates();
			}
			return false;
		});
	}

	if (playButton) {
		playButton.addEventListener('click', function(e) {
			e.preventDefault();
			if (activeRub()) {
				activeRub().play();
				isPlaying = true;
				updateButtonStates();
			}
			return false;
		});
	}

	if (restartButton) {
		restartButton.addEventListener('click', function(e) {
			e.preventDefault();
			if (activeRub()) {
				activeRub().move_to(0);
			}
			return false;
		});
	}

	if (previousButton) {
		previousButton.addEventListener('click', function(e) {
			e.preventDefault();
			if (activeRub()) {
				activeRub().move_relative(-1);
			}
			return false;
		});
	}

	if (nextButton) {
		nextButton.addEventListener('click', function(e) {
			e.preventDefault();
			if (activeRub()) {
				activeRub().move_relative(1);
			}
			return false;
		});
	}

	// Handle demo tab switching
	const tabs = document.querySelectorAll('.tabs#demo-tabs ul li a');
	const demoImages = document.querySelectorAll('.figure-demo');

	tabs.forEach((tab) => {
		tab.addEventListener('click', function(event) {
			event.preventDefault();
			if (tab.parentElement.classList.contains('is-active')) {
				restartButton.click(); // If already active, restart the GIF
				return; // Already active, do nothing
			}

			// Remove active class from all tabs
			tabs.forEach((t) => t.parentElement.classList.remove('is-active'));
			// Add active class to clicked tab
			tab.parentElement.classList.add('is-active');

			// Update demo images
			// fallback if rubbable failed
			demoImages.forEach((img) => {
				img.src = DEMOS[tab.parentElement.dataset.choose];
			});
			Object.values(rubs).forEach((rub) => {
				rub.pause();
				rub.get_canvas().style.display = 'none';
			});
			activeDemo = DEMOS[tab.parentElement.dataset.choose];
			activeRub().get_canvas().style.display = 'unset'; // Show the current GIF canvas
			pauseButton.click();
			restartButton.click(); // Restart the GIF for the new demo
			playButton.click(); // Play the GIF for the new demo
		});
	});

	// Initialize button states
	updateButtonStates();
});
