// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize any interactive elements
    initFormControlDependencies();
    
    // Add event listeners for form submissions
    setupFormValidation();
});

// Initialize form control dependencies based on selected options
function initFormControlDependencies() {
    // Only run on pages that have these elements
    const phoneServiceSelect = document.getElementById('PhoneService');
    const internetServiceSelect = document.getElementById('InternetService');
    
    if (phoneServiceSelect) {
        phoneServiceSelect.addEventListener('change', function() {
            const multipleLinesSelect = document.getElementById('MultipleLines');
            if (this.value === 'No') {
                multipleLinesSelect.value = 'No phone service';
                multipleLinesSelect.disabled = true;
            } else {
                multipleLinesSelect.disabled = false;
            }
        });
        
        // Trigger the event on page load
        phoneServiceSelect.dispatchEvent(new Event('change'));
    }
    
    if (internetServiceSelect) {
        internetServiceSelect.addEventListener('change', function() {
            const internetDependentServices = [
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies'
            ];
            
            if (this.value === 'No') {
                internetDependentServices.forEach(service => {
                    const select = document.getElementById(service);
                    if (select) {
                        select.value = 'No internet service';
                        select.disabled = true;
                    }
                });
            } else {
                internetDependentServices.forEach(service => {
                    const select = document.getElementById(service);
                    if (select) {
                        select.disabled = false;
                    }
                });
            }
        });
        
        // Trigger the event on page load
        internetServiceSelect.dispatchEvent(new Event('change'));
    }
    
    // Add automatic calculation of total charges
    const monthlyChargesInput = document.getElementById('MonthlyCharges');
    const tenureInput = document.getElementById('tenure');
    
    if (monthlyChargesInput && tenureInput) {
        const updateTotalCharges = function() {
            const totalChargesInput = document.getElementById('TotalCharges');
            const monthlyCharges = parseFloat(monthlyChargesInput.value) || 0;
            const tenure = parseInt(tenureInput.value) || 0;
            
            if (totalChargesInput) {
                totalChargesInput.value = (monthlyCharges * tenure).toFixed(2);
            }
        };
        
        monthlyChargesInput.addEventListener('input', updateTotalCharges);
        tenureInput.addEventListener('input', updateTotalCharges);
    }
}

// Setup form validation
function setupFormValidation() {
    const singlePredictionForm = document.querySelector('form[action="/single_prediction"]');
    const batchPredictionForm = document.querySelector('form[action="/batch_prediction"]');
    
    if (singlePredictionForm) {
        singlePredictionForm.addEventListener('submit', function(event) {
            // Basic validation for required fields
            const requiredFields = [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                'PhoneService', 'InternetService', 'Contract', 'PaperlessBilling',
                'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
            ];
            
            let isValid = true;
            
            requiredFields.forEach(field => {
                const input = document.getElementById(field);
                if (input && !input.value) {
                    input.classList.add('is-invalid');
                    isValid = false;
                } else if (input) {
                    input.classList.remove('is-invalid');
                }
            });
            
            if (!isValid) {
                event.preventDefault();
                alert('Please fill in all required fields.');
            }
        });
    }
    
    if (batchPredictionForm) {
        batchPredictionForm.addEventListener('submit', function(event) {
            const fileInput = document.getElementById('file');
            
            if (!fileInput || !fileInput.files.length) {
                event.preventDefault();
                alert('Please select a CSV file to upload.');
                return;
            }
            
            const file = fileInput.files[0];
            
            if (!file.name.endsWith('.csv')) {
                event.preventDefault();
                alert('Please upload a CSV file.');
                return;
            }
        });
    }
}
