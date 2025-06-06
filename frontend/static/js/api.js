/**
 * @file api.js
 * @description Manages all API communication with the backend server.
 * This module abstracts the details of HTTP requests for fetching data,
 * managing sessions, and other backend interactions.
 */

// --- API Configuration ---

// Dynamically determine the protocol and hostname for the API endpoint.
let API_PROTOCOL = window.location.protocol;
let API_HOSTNAME = window.location.hostname;
const API_PORT = '8000'; // The port where the FastAPI backend is running.

// If the frontend is opened as a local file, default to localhost for API calls.
// This is a crucial fallback for local development without a web server.
if (API_PROTOCOL === 'file:') {
    API_PROTOCOL = 'http:';
    API_HOSTNAME = 'localhost'; // Or '127.0.0.1'
    console.warn('Frontend is opened as a local file. API calls will be directed to http://localhost:8000.');
}

const API_BASE_URL = `${API_PROTOCOL}//${API_HOSTNAME}:${API_PORT}`;

// --- Session Management Functions ---

/**
 * Initiates a new user session with the backend.
 * @returns {Promise<Object>} A promise that resolves to the session data,
 * containing a unique `session_token`.
 */
function initiateSession() {
    return fetch(`${API_BASE_URL}/utils/session/initiate`).then(res => res.json());
}

/**
 * Sends a heartbeat to the backend to keep the current session alive.
 * This prevents session-specific cached data from being prematurely cleaned up.
 * @param {string} token - The user's current session token.
 * @returns {Promise<Object>} A promise that resolves to the heartbeat status.
 */
function sendHeartbeat(token) {
    return fetch(`${API_BASE_URL}/utils/session/heartbeat`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ session_token: token }),
    }).then(res => res.json());
}


// --- Historical Data Functions ---

/**
 * Constructs the URL for fetching the initial set of historical data.
 * @param {string} sessionToken - The unique session token for the user.
 * @param {string} exchange - The exchange name (e.g., "NASDAQ").
 * @param {string} token - The asset symbol/token (e.g., "AAPL").
 * @param {string} interval - The data interval (e.g., "1m", "1d").
 * @param {string} startTime - The start time in ISO format.
 * @param {string} endTime - The end time in ISO format.
 * @returns {string} The full API URL for the initial historical data request.
 */
function getHistoricalDataUrl(sessionToken, exchange, token, interval, startTime, endTime) {
    const params = new URLSearchParams({
        session_token: sessionToken,
        exchange: exchange,
        token: token,
        interval: interval,
        start_time: startTime,
        end_time: endTime
    });
    return `${API_BASE_URL}/historical/?${params.toString()}`;
}

/**
 * Constructs the URL for fetching a subsequent chunk of historical data.
 * This is used for pagination or lazy-loading older data into the chart.
 * @param {string} requestId - The unique ID for the data request session,
 * obtained from the initial data fetch.
 * @param {number} offset - The starting index (offset) of the data to fetch.
 * @param {number} [limit=5000] - The number of data points to fetch.
 * @returns {string} The full API URL for fetching a data chunk.
 */
function getHistoricalDataChunkUrl(requestId, offset, limit = 5000) {
    const params = new URLSearchParams({
        request_id: requestId,
        offset: offset,
        limit: limit
    });
    return `${API_BASE_URL}/historical/chunk?${params.toString()}`;
}