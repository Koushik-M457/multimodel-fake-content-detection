async function analyzeImage(img) {

    if (img.dataset.checked) return;
    img.dataset.checked = "true";

    try {

        const response = await fetch(img.src);
        const blob = await response.blob();

        const formData = new FormData();
        formData.append("image", blob, "image.jpg");
        formData.append("caption", "");
        formData.append("hashtags", "");

        const result = await fetch("http://127.0.0.1:8000/analyze", {
            method: "POST",
            body: formData
        });

        const data = await result.json();

        console.log("AI Detection:", data);

        if (data.verdict === "FAKE") {

            img.style.filter = "blur(20px)";

            const label = document.createElement("div");

            label.innerText = "⚠ AI GENERATED";

            label.style.position = "absolute";
            label.style.background = "red";
            label.style.color = "white";
            label.style.padding = "5px";
            label.style.fontWeight = "bold";
            label.style.zIndex = "9999";

            img.parentElement.style.position = "relative";
            img.parentElement.appendChild(label);
        }

    } catch (error) {
        console.log("Detection error", error);
    }
}

function scanImages() {

    const images = document.querySelectorAll("img");

    images.forEach(img => {

        if (img.width > 200 && img.height > 200) {
            analyzeImage(img);
        }

    });
}

setInterval(scanImages, 4000);