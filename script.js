// Hamburger Menu Toggle
const hamburger = document.getElementById('hamburger');
const navLinks = document.getElementById('navLinks');
hamburger.addEventListener('click', () => {
  navLinks.classList.toggle('active');
});

// Click to trigger file input
const uploadZone = document.getElementById('uploadContainer');
const fileInput = document.getElementById('fileInput');
uploadZone.addEventListener('click', () => {
  fileInput.click();
  document.getElementById("upload").scrollIntoView({ behavior: "smooth" });

  fileInput.addEventListener('change', () => {
    // Clear result
    resultDisplay.textContent = "";
  
    // Hide video preview and reset src
    videoPreviewContainer.style.display = "none";
    videoPreview.src = "";
  
    // Hide accuracy score
    accuracyScoreDisplay.style.display = "none";
    accuracyScoreDisplay.textContent = "";
  
    // Clear analysis steps animation
    steps.forEach(step => step.classList.remove('active'));
  
    // Clear stored values
    lastResult = '';
    lastConfidence = '';
  });
  
});

// Click to get video preview 
const videoPreviewCard = document.getElementById('videoPreviewCard');
const howItWorksSection = document.getElementById('how-it-works');
const videoPreviewContainer = document.getElementById('videoPreviewContainer');
const videoPreview = document.getElementById('videoPreview');
// When user clicks the "Video Preview" card
videoPreviewCard.addEventListener('click', () => {
  const videoFile = fileInput.files[0];

  if (!videoFile) {
    alert("Please upload a video file first.");
    return;
  }

  const videoURL = URL.createObjectURL(videoFile);
  videoPreview.src = videoURL;
  videoPreviewContainer.style.display = 'block';


  
 // 👉 Show the Video Preview title
 document.getElementById("videoPreviewTitle").style.display = "block";

  // Scroll to the "How it works" section
  howItWorksSection.scrollIntoView({ behavior: 'smooth' });
});

// Click to get upload Section
document.getElementById("uploadFeatureCard").addEventListener("click", () => {
  const uploadSection = document.getElementById("upload");
  uploadSection.style.display = "block"; // 👈 show section
  uploadSection.scrollIntoView({ behavior: "smooth" });
});

// Click to get upload Section using navlink
document.getElementById("uploadNavLink").addEventListener("click", () => {
  const uploadSection = document.getElementById("upload");
  uploadSection.style.display = "block"; // 👈 show section
  uploadSection.scrollIntoView({ behavior: "smooth" });
});

// Analyze Button Logic
const analyzeBtn = document.getElementById('analyzeBtn');
const analysisDemo = document.getElementById('analysisDemo');
const steps = document.querySelectorAll('.process-step');
const resultDisplay = document.getElementById('resultDisplay');
const accuracyScoreDisplay = document.getElementById('accuracyScoreDisplay');
const processOverviewCard = document.getElementById('processOverviewCard');
const accuracyScoreCard = document.getElementById('accuracyScoreCard');

let lastResult = '';
let lastConfidence = '';// Store last prediction result

analyzeBtn.addEventListener('click', () => {
  const videoFile = fileInput.files[0];
  if (!videoFile) {
    alert("Please upload a video file first.");
    return;
  }

  analyzeBtn.disabled = true;
  resultDisplay.textContent = "🔍 Analyzing video...";
  accuracyScoreDisplay.style.display = "none";

  // Start step-by-step animation
  analysisDemo.classList.add('active');
  steps.forEach((step) => step.classList.remove('active'));

  steps.forEach((step, index) => {
    setTimeout(() => {
      step.classList.add('active');
    }, index * 1000); // 1s delay between each step
  });

  // Upload after animation finishes (6 steps = 6s)
  setTimeout(() => {
    const formData = new FormData();
    formData.append("video", videoFile);

    fetch("/upload", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.label) {
          lastResult = data.label;
          lastConfidence = data.confidence;
          resultDisplay.textContent = "✅ Result: " + data.label;
        } else if (data.error) {
          resultDisplay.textContent = "❌ Error: " + data.error;
        }
      })
      .catch((error) => {
        console.error("Error:", error);
        resultDisplay.textContent = "❌ Error: Something went wrong.";
      })
      .finally(() => {
        analyzeBtn.disabled = false;
      });
  }, steps.length * 1000); // Wait for all animations
});

accuracyScoreCard.addEventListener('click', () => {
  if (lastConfidence !== '') {
    accuracyScoreDisplay.textContent = `🎯 Confidence Score: ${lastConfidence}%`;
    accuracyScoreDisplay.style.display = "block";

    const uploadSection = document.getElementById("upload");
    uploadSection.style.display = "block";
    uploadSection.scrollIntoView({ behavior: "smooth" });
  } else {
    alert("Please upload and analyze a video first.");
  }
});


processOverviewCard.addEventListener('click', () => {
  // Scroll to the "how-it-works" section
  howItWorksSection.scrollIntoView({ behavior: 'smooth' });

  // Remove previous active classes
  steps.forEach(step => step.classList.remove('active'));

  // Animate step-by-step
  steps.forEach((step, index) => {
    setTimeout(() => {
      step.classList.add('active');
    }, index * 1000); // 1 second delay per step
  });

});

const toggle = document.getElementById("themeSwitch");

toggle.addEventListener("change", () => {
  document.body.classList.toggle("dark-mode");
});



