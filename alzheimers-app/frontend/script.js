// ============================================
//   NeuroScan — script.js
//   Handles: Login, File Upload, API Call,
//            Preview, Result Display
// ============================================

const FLASK_API = 'http://127.0.0.1:5000';

// ─── Dummy credentials (replace with real auth later) ───
const VALID_USERS = {
  admin: '1234',
  doctor: 'neuro2024'
};

// ─── Stage metadata ───
const STAGE_META = {
  NonDemented: {
    label: 'Non-Demented',
    icon: '🟢',
    colorClass: 'stage-none',
    description: 'No signs of Alzheimer\'s disease detected. The MRI scan appears normal. Regular monitoring is still recommended as a precaution.'
  },
  VeryMildDemented: {
    label: 'Very Mild Demented',
    icon: '🟡',
    colorClass: 'stage-verymild',
    description: 'Very early signs of cognitive decline detected. Memory lapses may occur occasionally. Lifestyle modifications and regular follow-ups are advised.'
  },
  MildDemented: {
    label: 'Mild Demented',
    icon: '🟠',
    colorClass: 'stage-mild',
    description: 'Mild cognitive impairment observed. The patient may experience noticeable memory difficulties. Clinical evaluation and specialist consultation are recommended.'
  },
  ModerateDemented: {
    label: 'Moderate Demented',
    icon: '🔴',
    colorClass: 'stage-moderate',
    description: 'Moderate Alzheimer\'s progression detected. Significant memory and cognitive impairment present. Immediate medical attention and care planning are strongly advised.'
  }
};

// ============================================
//   LOGIN PAGE
// ============================================
function handleLogin() {
  const username = document.getElementById('username')?.value.trim();
  const password = document.getElementById('password')?.value;
  const errorEl  = document.getElementById('loginError');

  if (!username || !password) {
    errorEl.textContent = 'Please enter both username and password.';
    return;
  }

  if (VALID_USERS[username] && VALID_USERS[username] === password) {
    // Store session flag
    sessionStorage.setItem('neuroscan_user', username);
    errorEl.textContent = '';
    // Animate then redirect
    document.querySelector('.login-card').style.opacity = '0.5';
    document.querySelector('.login-card').style.transform = 'scale(0.97)';
    document.querySelector('.login-card').style.transition = 'all 0.3s ease';
    setTimeout(() => { window.location.href = 'upload.html'; }, 300);
  } else {
    errorEl.textContent = 'Invalid username or password.';
    // Shake animation
    const card = document.querySelector('.login-card');
    card.style.animation = 'shake 0.4s ease';
    setTimeout(() => { card.style.animation = ''; }, 400);
  }
}

// Allow Enter key on login
document.addEventListener('keydown', function(e) {
  if (e.key === 'Enter') {
    if (document.getElementById('username') || document.getElementById('password')) {
      handleLogin();
    }
  }
});

// ============================================
//   UPLOAD PAGE — Auth Guard
// ============================================
if (document.querySelector('.upload-page')) {
  if (!sessionStorage.getItem('neuroscan_user')) {
    window.location.href = 'index.html';
  }
}

function handleLogout() {
  sessionStorage.removeItem('neuroscan_user');
  window.location.href = 'index.html';
}

// ============================================
//   FILE HANDLING
// ============================================
let selectedFile = null;

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (!file) return;
  loadFile(file);
}

function loadFile(file) {
  // Validate type
  if (!file.type.startsWith('image/')) {
    alert('Please upload a valid image file (JPG, PNG, BMP).');
    return;
  }
  // Validate size (max 10MB)
  if (file.size > 10 * 1024 * 1024) {
    alert('File size must be under 10MB.');
    return;
  }

  selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById('previewImg').src = e.target.result;
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('previewBox').style.display = 'block';
    document.getElementById('dropZone').style.display = 'none';
    document.getElementById('analyzeBtn').disabled = false;
  };
  reader.readAsDataURL(file);
}

function clearFile() {
  selectedFile = null;
  document.getElementById('fileInput').value = '';
  document.getElementById('previewBox').style.display = 'none';
  document.getElementById('dropZone').style.display = 'flex';
  document.getElementById('analyzeBtn').disabled = true;
  resetResult();
}

// ─── Drag & Drop ───
document.addEventListener('DOMContentLoaded', () => {
  const dropZone = document.getElementById('dropZone');
  if (!dropZone) return;

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });
  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) loadFile(file);
  });
});

// ============================================
//   SUBMIT TO FLASK
// ============================================
async function submitImage() {
  if (!selectedFile) return;

  showState('loading');

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const response = await fetch(`${FLASK_API}/predict`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.error || `Server error: ${response.status}`);
    }

    const data = await response.json();
    if (data.prediction) {
      showResult(data.prediction);
    } else {
      throw new Error('Unexpected response from server.');
    }

  } catch (err) {
    showError(err.message || 'Could not connect to the backend. Is Flask running?');
  }
}

// ============================================
//   RESULT DISPLAY
// ============================================
function showResult(stageName) {
  const meta = STAGE_META[stageName] || {
    label: stageName,
    icon: '🔵',
    colorClass: '',
    description: 'Prediction received from the model.'
  };

  // Badge
  document.getElementById('badgeIcon').textContent = meta.icon;
  document.getElementById('badgeStage').textContent = meta.label;

  const badge = document.getElementById('resultBadge');
  badge.className = 'result-badge ' + meta.colorClass;

  // Description
  document.getElementById('stageDescription').textContent = meta.description;

  // Scale highlight
  document.querySelectorAll('.scale-step').forEach(step => {
    step.classList.remove('active');
    if (step.dataset.stage === stageName) {
      step.classList.add('active');
    }
  });

  showState('output');
}

function showError(msg) {
  document.getElementById('errorMessage').textContent = msg;
  showState('error');
}

function resetResult() {
  showState('idle');
}

function resetAnalysis() {
  clearFile();
  showState('idle');
}

function showState(state) {
  const states = ['idle', 'loading', 'output', 'error'];
  states.forEach(s => {
    const el = document.getElementById('result' + s.charAt(0).toUpperCase() + s.slice(1));
    if (el) el.style.display = (s === state) ? '' : 'none';
  });
}

// Add shake keyframes dynamically for login
const style = document.createElement('style');
style.textContent = `
  @keyframes shake {
    0%,100% { transform: translateX(0); }
    20%,60%  { transform: translateX(-8px); }
    40%,80%  { transform: translateX(8px); }
  }
`;
document.head.appendChild(style);
