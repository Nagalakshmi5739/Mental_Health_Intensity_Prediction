document.addEventListener('DOMContentLoaded', () => {
    const textInput = document.getElementById('textInput');
    const predictBtn = document.getElementById('predictBtn');
    const resetBtn = document.getElementById('resetBtn');
    const predictForm = document.getElementById('predictForm');
    const loading = document.getElementById('loading');
    const resultCard = document.getElementById('resultCard');
    const predictedClass = document.getElementById('predictedClass');
    const confidenceScore = document.getElementById('confidenceScore');
    const probabilities = document.getElementById('probabilities');
    const errorCard = document.getElementById('errorCard');
    const errorMessage = document.getElementById('errorMessage');

    const fileInput = document.getElementById('fileInput');
    const fileLabel = document.getElementById('fileLabel');
    const fileName = document.getElementById('fileName');
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadSection = document.getElementById('uploadSection');
    const batchResultCard = document.getElementById('batchResultCard');
    const batchTableBody = document.getElementById('batchTableBody');
    const downloadLink = document.getElementById('downloadLink');
    const totalCount = document.getElementById('totalCount');
    const correctCount = document.getElementById('correctCount');
    const accuracyPercent = document.getElementById('accuracyPercent');
    const dropZone = document.getElementById('dropZone');

    const validateInput = () => {
        const text = textInput.value.trim();
        if (text.length === 0) {
            showError('Please enter some text before predicting.');
            return false;
        }
        hideError();
        return true;
    };

    const showLoading = () => {
        loading.classList.remove('hidden');
        resultCard.classList.add('hidden');
        errorCard.classList.add('hidden');
        predictBtn.disabled = true;
    };

    const hideLoading = () => {
        loading.classList.add('hidden');
        predictBtn.disabled = false;
    };

    const showError = (msg) => {
        hideLoading();
        errorCard.classList.remove('hidden');
        resultCard.classList.add('hidden');
        errorMessage.textContent = msg;
    };

    const hideError = () => {
        errorCard.classList.add('hidden');
    };

    const setLoadingState = (shouldLoad) => {
        if (shouldLoad) {
            showLoading();
        } else {
            hideLoading();
        }
    };

    const buildProbabilityBars = (allProbabilities) => {
        probabilities.innerHTML = '';
        Object.entries(allProbabilities).forEach(([className, prob]) => {
            const probItem = document.createElement('div');
            probItem.className = 'prob-item';

            const probLabel = document.createElement('div');
            probLabel.className = 'prob-label';
            probLabel.textContent = className;

            const barBg = document.createElement('div');
            barBg.className = 'prob-bar-bg';

            const barFill = document.createElement('div');
            barFill.className = 'prob-bar-fill';
            barFill.style.width = '0%';

            const probValue = document.createElement('div');
            probValue.className = 'prob-value';
            probValue.textContent = `${prob}%`;

            barBg.appendChild(barFill);
            probItem.appendChild(probLabel);
            probItem.appendChild(barBg);
            probItem.appendChild(probValue);
            probabilities.appendChild(probItem);

            setTimeout(() => {
                barFill.style.width = `${Math.min(prob, 100)}%`;
            }, 50);
        });
    };

    const displayResult = (data) => {
        hideLoading();
        predictedClass.textContent = data.label || 'N/A';
        confidenceScore.textContent = data.confidence ? `${data.confidence}%` : 'N/A';

        predictedClass.style.backgroundColor =
            data.label === 'Very Negative'
                ? 'rgba(255, 71, 87, 0.25)'
                : data.label === 'Negative'
                ? 'rgba(255, 107, 107, 0.25)'
                : data.label === 'Neutral'
                ? 'rgba(255, 217, 61, 0.2)'
                : 'rgba(107, 255, 107, 0.2)';

        predictedClass.style.color =
            data.label === 'Very Negative'
                ? '#ff4757'
                : data.label === 'Negative'
                ? '#ff6b6b'
                : data.label === 'Neutral'
                ? '#ffd93d'
                : '#6bff6b';

        if (data.all_probabilities) {
            buildProbabilityBars(data.all_probabilities);
        }

        resultCard.classList.remove('hidden');
    };

    const resetForm = () => {
        textInput.value = '';
        resultCard.classList.add('hidden');
        errorCard.classList.add('hidden');
        hideLoading();
        textInput.focus();
    };

    resetBtn.addEventListener('click', resetForm);

    predictForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (!validateInput()) {
            return;
        }

        const text = textInput.value.trim();
        setLoadingState(true);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text_input: text }),
            });

            const data = await response.json();

            if (!response.ok) {
                showError(data.error || 'Failed to get prediction.');
                return;
            }

            displayResult(data);
        } catch (err) {
            console.error('Prediction error:', err);
            showError('Network error. Please try again later.');
            setLoadingState(false);
        }
    });

    if (fileInput) {
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                fileName.textContent = fileInput.files[0].name;
                uploadBtn.disabled = false;
            } else {
                fileName.textContent = 'Choose a CSV or Excel file';
                uploadBtn.disabled = true;
            }
        });
    }

    if (dropZone) {
        dropZone.addEventListener('click', () => {
            if (fileInput) {
                fileInput.click();
            }
        });

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.style.borderColor = 'var(--primary)';
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.style.borderColor = '';
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.style.borderColor = '';
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                fileName.textContent = e.dataTransfer.files[0].name;
                uploadBtn.disabled = false;
            }
        });
    }

    if (uploadBtn) {
        uploadBtn.addEventListener('click', async () => {
            if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
                showError('Please select a file first.');
                return;
            }

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            setLoadingState(true);

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData,
                });

                if (!response.ok) {
                    const errData = await response.json();
                    showError(errData.error || 'Batch prediction failed.');
                    return;
                }

                const blob = await response.blob();
                const csvUrl = URL.createObjectURL(blob);
                const fileName = 'batch_results.csv';

                if (downloadLink) {
                    downloadLink.href = csvUrl;
                    downloadLink.download = fileName;
                    downloadLink.style.display = 'inline-block';
                }

                const text = await blob.text();
                const lines = text.split('\n').filter((l) => l.trim());

                if (lines.length <= 1) {
                    showError('The uploaded file contains no data rows.');
                    return;
                }

                const headers = lines[0].split(',');
                const rows = lines.slice(1);
                const total = rows.length;
                const correct = rows.filter(
                    (row) => row.split(',')[3]?.trim().toLowerCase() === 'true'
                ).length;

                if (batchResultCard) {
                    batchResultCard.classList.remove('hidden');
                }

                if (totalCount) {
                    totalCount.textContent = total;
                }

                if (correctCount) {
                    correctCount.textContent = correct;
                }

                if (accuracyPercent) {
                    accuracyPercent.textContent =
                        ((correct / total) * 100).toFixed(1) + '%';
                }

                if (batchTableBody) {
                    batchTableBody.innerHTML = '';
                    rows.forEach((row) => {
                        const cols = row.split(',');
                        if (cols.length >= 4) {
                            const tr = document.createElement('tr');

                            const tdText = document.createElement('td');
                            tdText.textContent = cols[0] || '';
                            tdText.title = cols[0] || '';

                            const tdPred = document.createElement('td');
                            tdPred.textContent = cols[1] || '';
                            tdPred.style.fontWeight = '600';

                            const tdActual = document.createElement('td');
                            tdActual.textContent = cols[2] || '';

                            const tdCorrect = document.createElement('td');
                            tdCorrect.textContent = cols[3] || '';
                            tdCorrect.style.color =
                                cols[3]?.trim().toLowerCase() === 'true'
                                    ? '#6bff6b'
                                    : '#ff6b6b';
                            tdCorrect.style.fontWeight = '600';

                            tr.appendChild(tdText);
                            tr.appendChild(tdPred);
                            tr.appendChild(tdActual);
                            tr.appendChild(tdCorrect);
                            batchTableBody.appendChild(tr);
                        }
                    });
                }

                hideLoading();
            } catch (err) {
                console.error('Upload error:', err);
                showError('Network error during batch prediction.');
                setLoadingState(false);
            }
        });
    }
});
