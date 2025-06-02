// AudioContextの生成
const context = new AudioContext();

// サウンドの読み込み
const loadSound = (url) => {
    return new Promise((resolve) => {
        const request = new XMLHttpRequest();
        request.open('GET', url, true);
        request.responseType = 'arraybuffer';
        request.onload = () => {
            context.decodeAudioData(request.response, (buffer) => {
                resolve(buffer);
            });
        };
        request.send();
    });
};

// サウンドの再生用の変数を定義
let source = null;

// サウンドの再生
const playSound = (buffer) => {
    // Source
    source = context.createBufferSource();
    source.buffer = buffer;
    // Destination
    source.connect(context.destination);
    // Sourceの再生
    source.start(0);
};

// サウンドの停止
const stopSound = () => {
    if (source) {
        source.stop();
        source = null; // ソースをクリアして次回再生可能にする
    }
};

// 再生ボタンの処理
const startButton = document.getElementById('startButton');
startButton.addEventListener('click', async () => {
    if (context.state === 'suspended') {
        await context.resume();
    }
    try {
        const buffer = await loadSound('./music/shining_star.mp3');
        playSound(buffer);
    } catch (err) {
        console.error(err);
    }
});

// 停止ボタンの処理
const stopButton = document.getElementById('stopButton');
stopButton.addEventListener('click', () => {
    stopSound();
});
