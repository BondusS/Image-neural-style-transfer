document.addEventListener('DOMContentLoaded', function() {
    // Preview for content image
    const contentInput = document.getElementById('content_image');
    const contentPreview = document.createElement('div');
    contentPreview.className = 'image-preview';
    contentPreview.innerHTML = `
        <div class="image-preview-label">
            <i class="bi bi-image"></i>
            <span>Content Image Preview</span>
        </div>
        <img id="content-preview-img" />
    `;
    contentInput.parentNode.insertBefore(contentPreview, contentInput.nextSibling);
    
    // Preview for style image
    const styleInput = document.getElementById('style_image');
    const stylePreview = document.createElement('div');
    stylePreview.className = 'image-preview';
    stylePreview.innerHTML = `
        <div class="image-preview-label">
            <i class="bi bi-palette"></i>
            <span>Style Image Preview</span>
        </div>
        <img id="style-preview-img" />
    `;
    styleInput.parentNode.insertBefore(stylePreview, styleInput.nextSibling);
    
    // Handle file selection
    contentInput.addEventListener('change', function(e) {
        previewImage(e.target, 'content-preview-img', contentPreview);
    });
    
    styleInput.addEventListener('change', function(e) {
        previewImage(e.target, 'style-preview-img', stylePreview);
    });
    
    function previewImage(input, imgId, previewContainer) {
        const file = input.files[0];
        const img = previewContainer.querySelector(`#${imgId}`);
        
        if (file) {
            const reader = new FileReader();
            
            reader.addEventListener('load', function() {
                img.src = reader.result;
                previewContainer.classList.add('ready');
            });
            
            reader.readAsDataURL(file);
        } else {
            img.src = '';
            previewContainer.classList.remove('ready');
        }
    }
    
    // Add loading animation when form is submitted
    const form = document.querySelector('form');
    form.addEventListener('submit', function() {
        const submitBtn = form.querySelector('button[type="submit"]');
        submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Processing...';
        submitBtn.disabled = true;
    });
});