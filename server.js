const express = require("express");
const cors = require("cors");
const { pipeline } = require("@xenova/transformers");
const wordData = require("./data/words.json");

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 3001;
const ROUND_TIME = 5 * 60 * 1000;
const RESET_DELAY = 5000;

const MIN_SEMANTIC_CONFIDENCE = 0.80;

const serverData = {};
let embedder;

// ===============================
// EMBEDDING CACHE
// ===============================
const embeddingCache = new Map();

app.post("/", (req, res) => {
    res.json({ message: "Root working" });
});

// ===============================
// LOAD MODEL
// ===============================
async function loadModel() {
    console.log("⏳ Loading Embedding Model...");
    embedder = await pipeline(
        "feature-extraction",
        "Xenova/multilingual-e5-base"
    );
    console.log("✅ Model Loaded Successfully!");
}

// ===============================
// GET EMBEDDING (CACHE)
// ===============================
async function getEmbedding(text) {
    if (embeddingCache.has(text)) {
        return embeddingCache.get(text);
    }

    const result = await embedder("query: " + text, {
        pooling: "mean",
        normalize: true
    });

    embeddingCache.set(text, result.data);
    return result.data;
}

// ===============================
// COSINE SIMILARITY
// ===============================
function cosineSimilarity(a, b) {
    const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dot / (normA * normB);
}

// ===============================
// RANDOM WORD
// ===============================
function getRandomWordIndex() {
    return Math.floor(Math.random() * wordData.length);
}

// ===============================
// BUILD SECRET EMBEDDING
// ===============================
async function buildSecretEmbedding(wordObj) {
    const contextSentence =
        `คำนี้เกี่ยวกับ ${wordObj.relatedWords.join(", ")} และมีความหมายว่า ${wordObj.secretWord}`;

    return await getEmbedding(contextSentence);
}

// ===============================
// HYBRID VALIDATION
// ===============================
function isValidThaiStructure(word) {
    if (!word) return false;
    if (!/[ก-ฮ]/.test(word)) return false;
    if (word.length < 2) return false;
    if (/^(.)\1+$/.test(word)) return false;
    return true;
}

// ===============================
// CREATE SERVER
// ===============================
async function createServer(id) {
    const wordIndex = getRandomWordIndex();
    const currentWord = wordData[wordIndex];

    const secretEmbedding = await buildSecretEmbedding(currentWord);

    serverData[id] = {
        wordIndex,
        secretEmbedding,
        usedGuesses: {},
        leaderboard: {},
        startTime: Date.now(),
        timer: null,
        roundActive: true
    };

    startTimer(id);
    console.log("🆕 Server:", id, "| Answer:", currentWord.secretWord);
}

async function getServer(serverId) {
    const id = serverId && serverId !== "" ? serverId : "studio-test";
    if (!serverData[id]) {
        await createServer(id);
    }
    return serverData[id];
}

// ===============================
// TIMER
// ===============================
function startTimer(serverId) {
    const server = serverData[serverId];

    if (server.timer) clearTimeout(server.timer);

    server.startTime = Date.now();

    server.timer = setTimeout(async () => {
        server.roundActive = false;

        setTimeout(async () => {
            await newWord(serverId);
        }, RESET_DELAY);

    }, ROUND_TIME);
}

// ===============================
// NEW WORD
// ===============================
async function newWord(serverId) {
    const server = serverData[serverId];
    const newIndex = getRandomWordIndex();
    const currentWord = wordData[newIndex];

    const secretEmbedding = await buildSecretEmbedding(currentWord);

    server.wordIndex = newIndex;
    server.secretEmbedding = secretEmbedding;
    server.usedGuesses = {};
    server.leaderboard = {};
    server.roundActive = true;

    startTimer(serverId);

    console.log("🔄 New Answer:", currentWord.secretWord);
}

// ===============================
// CALCULATE SCORE (HYBRID V8)
// ===============================
async function calculatePercentageAI(server, guess, currentWord) {

    const secret = currentWord.secretWord.toLowerCase();
    const related = currentWord.relatedWords.map(w => w.toLowerCase());

    if (guess === secret) return 100;

    // Layer 1: Structure Validation
    if (!isValidThaiStructure(guess)) {
        return -1;
    }

    // Layer 2: Related Exact
    if (related.includes(guess)) {
        return parseFloat((92 + Math.random() * 5).toFixed(2));
    }

    // Layer 3: Semantic Check
    const guessEmbedding = await getEmbedding(
        `คำที่มีความหมายว่า ${guess}`
    );

    const semanticRaw = cosineSimilarity(
        guessEmbedding,
        server.secretEmbedding
    );

    if (semanticRaw < MIN_SEMANTIC_CONFIDENCE) {
        return -1;
    }

    const MIN_BASE = 0.82;

    let semanticScaled =
        (semanticRaw - MIN_BASE) / (1 - MIN_BASE);

    semanticScaled = Math.max(0, semanticScaled);
    semanticScaled = Math.pow(semanticScaled, 1.4);

    let finalScore = semanticScaled * 100;

    finalScore = Math.max(finalScore, 0.01);
    finalScore = Math.min(finalScore, 99.99);

    return parseFloat(finalScore.toFixed(2));
}

// ===============================
// GUESS ENDPOINT
// ===============================
app.post("/guess", async (req, res) => {

    const { serverId, guess, playerName } = req.body;

    if (!guess || !playerName) {
        return res.status(400).json({ error: "Missing data" });
    }

    const server = await getServer(serverId);

    if (!server.roundActive) {
        return res.json({ waiting: true });
    }

    const lowerGuess = guess.trim().toLowerCase();
    const currentWord = wordData[server.wordIndex];
    const lowerSecret = currentWord.secretWord.toLowerCase();

    if (server.usedGuesses[lowerGuess]) {
        return res.json({
            duplicate: true,
            by: server.usedGuesses[lowerGuess].playerName,
            percentage: server.usedGuesses[lowerGuess].percentage
        });
    }

    const percentage = await calculatePercentageAI(
        server,
        lowerGuess,
        currentWord
    );

    if (percentage === -1) {
        return res.json({
            correct: false,
            unknown: true,
            percentage: 0
        });
    }

    server.usedGuesses[lowerGuess] = {
        playerName,
        percentage
    };

    if (
        !server.leaderboard[playerName] ||
        percentage > server.leaderboard[playerName]
    ) {
        server.leaderboard[playerName] = percentage;
    }

    if (lowerGuess === lowerSecret) {
        server.roundActive = false;

        setTimeout(async () => {
            await newWord(serverId);
        }, RESET_DELAY);

        return res.json({
            correct: true,
            answer: currentWord.secretWord,
            winner: playerName,
            percentage: 100
        });
    }

    return res.json({
        correct: false,
        percentage
    });
});

// ===============================
// CURRENT ENDPOINT
// ===============================
app.post("/current", async (req, res) => {

    const { serverId } = req.body;
    const server = await getServer(serverId);

    const timeLeft = Math.max(
        0,
        ROUND_TIME - (Date.now() - server.startTime)
    );

    const currentWord = wordData[server.wordIndex];

    res.json({
        answer: currentWord.secretWord,
        timeLeft: Math.floor(timeLeft / 1000),
        leaderboard: server.leaderboard,
        guesses: server.usedGuesses,
        active: server.roundActive
    });
});

// ===============================
// START SERVER
// ===============================
loadModel().then(() => {
    app.listen(PORT, () => {
        console.log("🚀 THAI ULTRA AI V8 RUNNING ON " + PORT);
    });
});
{
  "name": "kuy",
  "version": "1.0.0",
  "description": "",
  "main": "server.js",
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1",
    "start": "node server.js"
  },
  "keywords": [],
  "author": "",
  "license": "ISC",
  "type": "commonjs",
  "dependencies": {
    "@xenova/transformers": "^2.17.2",
    "cors": "^2.8.6",
    "express": "^5.2.1",
    "string-similarity": "^4.0.4"
  }
}
