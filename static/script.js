document.addEventListener("DOMContentLoaded", () => {
  // --- Constants and UI Elements ---
  const CONSTANTS = {
    MAX_FILE_SIZE: 5 * 1024 * 1024, // 5MB
    ALLOWED_FILE_TYPES: ["image/jpeg", "image/png", "image/jpg"],
  };
  const ui = {
    dropzone: document.getElementById("dropzone"),
    processingState: document.getElementById("processingState"),
    resultsState: document.getElementById("resultsState"),
    fileInput: document.getElementById("fileInput"),
    selectBtn: document.getElementById("selectBtn"),
    resetBtn: document.getElementById("resetBtn"),
    resultImage: document.getElementById("resultImage"),
    predictionResult: document.getElementById("predictionResult"),
    resultsTitle: document.getElementById("resultsTitle"),
    memberMenu: document.getElementById("memberMenu"),
  };

  // --- LOGIC FOR MEMBER MENU ---
  if (ui.memberMenu) {
    const memberLabel = ui.memberMenu.querySelector(".member-label");
    memberLabel.addEventListener("click", (e) => {
      e.stopPropagation();
      ui.memberMenu.classList.toggle("active");
    });
    window.addEventListener("click", () => {
      if (ui.memberMenu.classList.contains("active")) {
        ui.memberMenu.classList.remove("active");
      }
    });
  }

  // --- LOGIC FOR FILE UPLOAD AND RESULT DISPLAY ---
  const showSection = (sectionToShow) => {
    [ui.dropzone, ui.processingState, ui.resultsState].forEach((section) => {
      section.classList.toggle("hidden", section !== sectionToShow);
    });
  };

  ui.selectBtn.addEventListener("click", () => ui.fileInput.click());

  ui.dropzone.addEventListener("click", (e) => {
    if (e.target.id !== "selectBtn" && !e.target.closest("#selectBtn")) {
      ui.fileInput.click();
    }
  });

  ui.fileInput.addEventListener("change", (e) =>
    handleFileSelect(e.target.files)
  );
  ui.dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    ui.dropzone.classList.add("dragover");
  });
  ui.dropzone.addEventListener("dragleave", () =>
    ui.dropzone.classList.remove("dragover")
  );
  ui.dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    ui.dropzone.classList.remove("dragover");
    handleFileSelect(e.dataTransfer.files);
  });

  ui.resetBtn.addEventListener("click", () => {
    ui.fileInput.value = "";
    showSection(ui.dropzone);
  });

  function handleFileSelect(files) {
    if (files.length === 0) return;
    const file = files[0];
    if (file.size > CONSTANTS.MAX_FILE_SIZE) {
      alert(
        `File is too large. Maximum size is ${
          CONSTANTS.MAX_FILE_SIZE / 1024 / 1024
        }MB.`
      );
      return;
    }
    if (!CONSTANTS.ALLOWED_FILE_TYPES.includes(file.type)) {
      alert("Invalid file format. Please select a JPG or PNG image.");
      return;
    }
    processFile(file);
  }

  function processFile(file) {
    showSection(ui.processingState);
    const formData = new FormData();
    formData.append("file", file);
    fetch("/predict", {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        if (!response.ok)
          throw new Error(`Server Error: ${response.statusText}`);
        return response.json();
      })
      .then((data) => {
        if (data.error) throw new Error(data.error);
        displayResults(file, data);
      })
      .catch((error) => {
        console.error("Prediction error:", error);
        alert(`An error occurred: ${error.message}`);
        showSection(ui.dropzone);
      });
  }

  function displayResults(file, data) {
    const reader = new FileReader();
    reader.onload = (e) => {
      ui.resultImage.src = e.target.result;
      ui.predictionResult.innerHTML = ""; // Clear old results

      const resultText = document.createElement("p");
      resultText.style.fontSize = "1.6rem";
      resultText.style.fontWeight = "700";
      resultText.style.color = "#4f46e5";

      // --- CẬP NHẬT VĂN BẢN KẾT QUẢ ĐỂ HIỂN THỊ MỆNH GIÁ TIỀN ---
      resultText.innerHTML = `AI predicts this is: <strong>${data.prediction} VND</strong>`;

      ui.predictionResult.appendChild(resultText);

      showSection(ui.resultsState);
    };
    reader.readAsDataURL(file);
  }
});
