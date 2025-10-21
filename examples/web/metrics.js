(function () {
  const wasm_promise = fetch("metrics.wasm");
  let wasm_exports = null;

  const canvasRef = document.getElementById("reference-canvas");
  const canvasDist = document.getElementById("distorted-canvas");
  const ctxRef = canvasRef.getContext("2d", { willReadFrequently: true });
  const ctxDist = canvasDist.getContext("2d", { willReadFrequently: true });
  const computeButton = document.getElementById("compute-button");
  const statusEl = document.getElementById("status");
  const psnrEl = document.getElementById("psnr-value");
  const ssimEl = document.getElementById("ssim-value");

  let refImageLoaded = false;
  let distImageLoaded = false;

  function clearResults() {
    psnrEl.textContent = "--";
    ssimEl.textContent = "--";
    statusEl.textContent = "";
  }

  function displayImage(canvas, ctx, file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = function (e) {
        const img = new Image();
        img.onload = function () {
          const maxSize = 2048;
          let width = img.width;
          let height = img.height;
          if (width > maxSize || height > maxSize) {
            const scale = Math.min(maxSize / width, maxSize / height);
            width = Math.floor(width * scale);
            height = Math.floor(height * scale);
          }

          canvas.width = width;
          canvas.height = height;
          ctx.drawImage(img, 0, 0, width, height);
          clearResults();
          resolve();
        };
        img.onerror = function () {
          reject(new Error("Failed to decode image."));
        };
        img.src = e.target.result;
      };
      reader.onerror = function () {
        reject(new Error("Failed to read file."));
      };
      reader.readAsDataURL(file);
    });
  }

  function makeFileInput(onFile) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.style.display = "none";
    input.addEventListener("change", function (event) {
      const file = event.target.files[0];
      if (file) onFile(file);
    });
    document.body.appendChild(input);
    return input;
  }

  function createLoadHandler(canvas, ctx, setLoaded) {
    return function (file) {
      setLoaded(false);
      displayImage(canvas, ctx, file)
        .then(function () {
          setLoaded(true);
        })
        .catch(function (error) {
          statusEl.textContent = error.message;
        });
    };
  }

  const handleRefFile = createLoadHandler(canvasRef, ctxRef, function (loaded) {
    refImageLoaded = loaded;
  });
  const handleDistFile = createLoadHandler(canvasDist, ctxDist, function (loaded) {
    distImageLoaded = loaded;
  });

  const refInput = makeFileInput(handleRefFile);
  const distInput = makeFileInput(handleDistFile);

  function enableDrop(canvas, { onClick, onDrop }) {
    canvas.addEventListener("click", onClick);
    canvas.addEventListener("dragover", function (event) {
      event.preventDefault();
    });
    canvas.addEventListener("drop", function (event) {
      event.preventDefault();
      const file = event.dataTransfer.files[0];
      if (file) onDrop(file);
    });
  }

  enableDrop(canvasRef, {
    onClick: function () {
      refInput.click();
    },
    onDrop: handleRefFile,
  });

  enableDrop(canvasDist, {
    onClick: function () {
      distInput.click();
    },
    onDrop: handleDistFile,
  });

  WebAssembly.instantiateStreaming(wasm_promise, {
    js: {
      log: function (ptr, len) {
        const view = new Uint8Array(wasm_exports.memory.buffer, ptr, len);
        const text = new TextDecoder().decode(view);
        console.log(text);
      },
      now: function () {
        return performance.now();
      },
    },
  }).then(function (obj) {
    wasm_exports = obj.instance.exports;
    window.wasm = obj;
    console.log("metrics wasm loaded");
  });

  computeButton.addEventListener("click", function () {
    if (!refImageLoaded || !distImageLoaded) {
      statusEl.textContent = "Load both images first.";
      return;
    }

    if (!wasm_exports) {
      statusEl.textContent = "WebAssembly module not ready yet.";
      return;
    }

    if (canvasRef.width !== canvasDist.width || canvasRef.height !== canvasDist.height) {
      statusEl.textContent = "Images must have the same dimensions.";
      return;
    }

    const width = canvasRef.width;
    const height = canvasRef.height;
    const pixelBytes = width * height * 4;

    const refData = ctxRef.getImageData(0, 0, width, height);
    const distData = ctxDist.getImageData(0, 0, width, height);

    const refPtr = wasm_exports.alloc(pixelBytes);
    const distPtr = wasm_exports.alloc(pixelBytes);
    const resultPtr = wasm_exports.alloc(8 * 2);

    const refBuffer = new Uint8Array(wasm_exports.memory.buffer, refPtr, pixelBytes);
    const distBuffer = new Uint8Array(wasm_exports.memory.buffer, distPtr, pixelBytes);
    refBuffer.set(refData.data);
    distBuffer.set(distData.data);

    const start = performance.now();
    wasm_exports.compute_metrics(refPtr, height, width, distPtr, height, width, resultPtr);
    const elapsed = performance.now() - start;

    const results = new Float64Array(wasm_exports.memory.buffer, resultPtr, 2);
    psnrEl.textContent = Number.isFinite(results[0]) ? results[0].toFixed(3) + " dB" : "--";
    ssimEl.textContent = Number.isFinite(results[1]) ? results[1].toFixed(6) : "--";
    statusEl.textContent = `time: ${elapsed.toFixed(1)} ms`;

    wasm_exports.free(refPtr, pixelBytes);
    wasm_exports.free(distPtr, pixelBytes);
    wasm_exports.free(resultPtr, 8 * 2);
  });
})();
