/**
 * @file main.js
 * @description Core logic for the stock chart frontend.
 * This script initializes the charting library, manages user interactions,
 * handles data fetching from the API, and updates the UI accordingly.
 * It includes session management, theme switching, and lazy-loading for chart data.
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- Element References ---
    const chartContainer = document.getElementById('chartContainer');
    const loadChartBtn = document.getElementById('loadChartBtn');
    const exchangeSelect = document.getElementById('exchange');
    const symbolSelect = document.getElementById('symbol');
    const intervalSelect = document.getElementById('interval');
    const startTimeInput = document.getElementById('start_time');
    const endTimeInput = document.getElementById('end_time');
    const themeToggle = document.getElementById('theme-toggle');
    const dataSummaryElement = document.getElementById('dataSummary');
    const loadingIndicator = document.getElementById('loadingIndicator');

    // --- State Management ---
    let allChartData = [];         // Holds all OHLC data currently loaded in the chart.
    let allVolumeData = [];        // Holds all volume data currently loaded.
    let currentlyFetching = false; // A flag to prevent concurrent data fetches.
    let allDataLoaded = false;     // A flag to indicate if all historical data has been loaded.

    // Pagination state for fetching data in chunks.
    let chartRequestId = null;     // The ID for the current chart data request session.
    let chartCurrentOffset = 0;    // The current offset of the data loaded.
    const DATA_CHUNK_SIZE = 5000;  // The number of data points to fetch per chunk.

    // Chart and session state.
    let mainChart = null;          // The main Lightweight Charts instance.
    let candleSeries = null;       // The candlestick series on the chart.
    let volumeSeries = null;       // The volume histogram series on the chart.
    let sessionToken = null;       // The user's session token.
    let heartbeatIntervalId = null;// The interval ID for the session heartbeat.

    /**
     * @function startSession
     * @description Initiates a session with the backend and starts a periodic heartbeat
     * to keep the session alive.
     */
    async function startSession() {
        try {
            const sessionData = await initiateSession();
            if (sessionData && sessionData.session_token) {
                sessionToken = sessionData.session_token;
                console.log(`Session started with token: ${sessionToken}`);
                showToast(`Session started.`, 'info');
                
                // Clear any existing heartbeat interval before starting a new one.
                if (heartbeatIntervalId) clearInterval(heartbeatIntervalId);

                // Send a heartbeat every 60 seconds.
                heartbeatIntervalId = setInterval(async () => {
                    if (sessionToken) {
                        try {
                            const heartbeatStatus = await sendHeartbeat(sessionToken);
                            if (heartbeatStatus.status !== 'ok') {
                                console.error('Heartbeat failed:', heartbeatStatus.message);
                                clearInterval(heartbeatIntervalId);
                                showToast('Session expired. Please reload the page.', 'error');
                            } else {
                                console.log('Heartbeat sent successfully.');
                            }
                        } catch (e) {
                            console.error('Error sending heartbeat:', e);
                            clearInterval(heartbeatIntervalId);
                            showToast('Connection lost. Please reload.', 'error');
                        }
                    }
                }, 60000); // 60 seconds
            } else {
                throw new Error("Invalid session data received from server.");
            }
        } catch (error) {
            console.error('Failed to initiate session:', error);
            showToast('Could not start a session. Please check connection and reload.', 'error');
        }
    }

    /**
     * @function getChartTheme
     * @description Gets the appropriate theme options for the chart based on the
     * current HTML data-theme attribute (light or dark).
     * @returns {object} A configuration object for the Lightweight Charts library.
     */
    const getChartTheme = () => {
        const isDarkMode = document.documentElement.getAttribute('data-theme') === 'dark';
        return {
            layout: { 
                background: { type: 'solid', color: isDarkMode ? '#1d232a' : '#ffffff' }, 
                textColor: isDarkMode ? '#a6adba' : '#1f2937' 
            },
            grid: { 
                vertLines: { color: isDarkMode ? '#2a323c' : '#e5e7eb' }, 
                horzLines: { color: isDarkMode ? '#2a323c' : '#e5e7eb' } 
            },
            crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
            rightPriceScale: { borderColor: isDarkMode ? '#2a323c' : '#e5e7eb' },
            timeScale: { 
                borderColor: isDarkMode ? '#2a323c' : '#e5e7eb', 
                timeVisible: true, 
                secondsVisible: ['1s', '5s', '10s', '15s', '30s', '45s'].includes(intervalSelect.value) 
            },
        };
    };
    
    /**
     * @function initializeCharts
     * @description Creates or re-creates the Lightweight Chart instance with the
     * correct theme and series. It also sets up the lazy-loading
     * mechanism for historical data.
     */
    function initializeCharts() {
        if (mainChart) mainChart.remove(); // Remove old chart instance if it exists.
        
        mainChart = LightweightCharts.createChart(chartContainer, getChartTheme());
        candleSeries = mainChart.addCandlestickSeries({ upColor: '#10b981', downColor: '#ef4444', borderVisible: false, wickUpColor: '#10b981', wickDownColor: '#ef4444' });
        
        // Add volume series to a separate pane below the main chart.
        volumeSeries = mainChart.addHistogramSeries({ color: '#9ca3af', priceFormat: { type: 'volume' }, priceScaleId: '' });
        mainChart.priceScale('').applyOptions({ scaleMargins: { top: 0.7, bottom: 0 } });

        // Subscribe to the visible range change event for lazy loading.
        mainChart.timeScale().subscribeVisibleLogicalRangeChange(async (newVisibleRange) => {
            if (!newVisibleRange || currentlyFetching || allDataLoaded || !chartRequestId) {
                return;
            }

            // Trigger fetch for older data when the user scrolls near the beginning of the current data.
            const lazyLoadThreshold = 10;
            if (newVisibleRange.from < lazyLoadThreshold) {
                await fetchAndPrependDataChunk();
            }
        });
    }
    
    /**
     * @function fetchAndPrependDataChunk
     * @description Fetches the next chunk of older historical data from the server
     * and prepends it to the existing chart data.
     */
    async function fetchAndPrependDataChunk() {
        if(currentlyFetching || allDataLoaded) return;
        currentlyFetching = true;
        loadingIndicator.style.display = 'flex';
        showToast('Loading older data...', 'info');
        
        // The API expects the offset of the *start* of the chunk we want.
        // Since we are prepending, the next offset will be the current one minus the chunk size.
        const nextOffset = chartCurrentOffset - DATA_CHUNK_SIZE;
        if (nextOffset < 0) {
            allDataLoaded = true; // No more data to fetch before the beginning.
            loadingIndicator.style.display = 'none';
            currentlyFetching = false;
            showToast('All available historical data loaded.', 'success');
            return;
        }

        const apiUrl = getHistoricalDataChunkUrl(chartRequestId, nextOffset, DATA_CHUNK_SIZE);

        try {
            const response = await fetch(apiUrl);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: response.statusText }));
                throw new Error(`HTTP error ${response.status}: ${errorData.detail || 'Failed to fetch chunk'}`);
            }
            
            const chunkData = await response.json();
            
            if (!chunkData || !Array.isArray(chunkData.candles) || chunkData.candles.length === 0) {
                allDataLoaded = true;
                console.log("Lazy loading complete: No more older candles were returned.");
                return;
            }

            const chartFormattedData = chunkData.candles.map(item => ({ time: item.unix_timestamp, open: item.open, high: item.high, low: item.low, close: item.close }));
            const volumeFormattedData = chunkData.candles.map(item => ({ time: item.unix_timestamp, value: item.volume, color: item.close > item.open ? 'rgba(16, 185, 129, 0.5)' : 'rgba(239, 68, 68, 0.5)' }));

            // Prepend the new (older) data to our existing data arrays.
            allChartData = [...chartFormattedData, ...allChartData];
            allVolumeData = [...volumeFormattedData, ...allVolumeData];
            
            // Update the offset to the new beginning of our data.
            chartCurrentOffset = chunkData.offset;
            if (chartCurrentOffset === 0) {
                allDataLoaded = true;
            }

            // Update the chart with the full dataset.
            candleSeries.setData(allChartData);
            volumeSeries.setData(allVolumeData);

            showToast(`Older data loaded. Total points: ${allChartData.length}`, 'success');
        } catch (error) {
            console.error('Failed to fetch older chart data:', error);
            showToast(`Error: ${error.message}`, 'error');
        } finally {
            loadingIndicator.style.display = 'none';
            currentlyFetching = false;
        }
    }

    /**
     * @function updateDataSummary
     * @description Updates the data summary panel below the chart with the latest
     * OHLCV data.
     * @param {object} latestData - The latest candle data object.
     * @param {string} symbol - The current symbol.
     * @param {string} exchange - The current exchange.
     * @param {string} interval - The current interval.
     */
    function updateDataSummary(latestData, symbol, exchange, interval) {
        if (!latestData) {
            dataSummaryElement.innerHTML = 'No data to summarize.';
            return;
        }
        const change = latestData.close - latestData.open;
        const changePercent = (latestData.open === 0) ? 0 : (change / latestData.open) * 100;
        const changeClass = change >= 0 ? 'text-success' : 'text-error';
        const dateObj = new Date(latestData.time * 1000);
        const formattedDate = `${dateObj.getDate().toString().padStart(2, '0')}/${(dateObj.getMonth() + 1).toString().padStart(2, '0')}/${dateObj.getFullYear()} ${dateObj.getHours().toString().padStart(2, '0')}:${dateObj.getMinutes().toString().padStart(2, '0')}:${dateObj.getSeconds().toString().padStart(2, '0')}`;
        const lastVolumeData = allVolumeData.find(d => d.time === latestData.time);
        const volume = lastVolumeData ? lastVolumeData.value.toLocaleString() : 'N/A';
        dataSummaryElement.innerHTML = `
            <strong>${symbol} (${exchange}) - ${interval}</strong><br>
            Last: O: ${latestData.open.toFixed(2)} H: ${latestData.high.toFixed(2)} L: ${latestData.low.toFixed(2)} C: ${latestData.close.toFixed(2)} V: ${volume}<br>
            Change: <span class="${changeClass}">${change.toFixed(2)} (${changePercent.toFixed(2)}%)</span><br>
            Time: ${formattedDate}
        `;
    }

    /**
     * @function applyTheme
     * @description Applies a theme (light or dark) to the application and chart.
     * @param {string} theme - The theme to apply ('light' or 'dark').
     */
    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('chartTheme', theme); // Persist theme choice.
        if (mainChart) mainChart.applyOptions(getChartTheme());
    }

    /**
     * @function setDefaultDateTime
     * @description Sets default start and end date/time values in the input fields.
     */
    function setDefaultDateTime() {
        const now = new Date();
        const oneMonthAgo = new Date(now);
        oneMonthAgo.setMonth(now.getMonth() - 1);
        oneMonthAgo.setHours(0, 0, 0, 0); 
        
        const endDateTime = new Date(now);
        endDateTime.setHours(23,59,59,999); 
        
        const formatForInput = (date) => {
            const year = date.getFullYear();
            const month = (date.getMonth() + 1).toString().padStart(2, '0');
            const day = date.getDate().toString().padStart(2, '0');
            const hours = date.getHours().toString().padStart(2, '0');
            const minutes = date.getMinutes().toString().padStart(2, '0');
            return `${year}-${month}-${day}T${hours}:${minutes}`;
        };

        startTimeInput.value = formatForInput(oneMonthAgo);
        endTimeInput.value = formatForInput(endDateTime);
    }
    
    /**
     * @function showToast
     * @description Displays a temporary notification message (toast).
     * @param {string} message - The message to display.
     * @param {string} [type='info'] - The type of toast ('info', 'success', 'warning', 'error').
     */
    function showToast(message, type = 'info') {
        const toastContainer = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `alert alert-${type} shadow-lg animate-pulse`;
        toast.style.animationDuration = '2s';
        
        // Icon selection based on toast type
        let iconHtml = '';
        if (type === 'success') iconHtml = '<svg xmlns="http://www.w3.org/2000/svg" class="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>';
        else if (type === 'error') iconHtml = '<svg xmlns="http://www.w3.org/2000/svg" class="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>';
        else if (type === 'warning') iconHtml = '<svg xmlns="http://www.w3.org/2000/svg" class="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>';
        else iconHtml = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" class="stroke-info shrink-0 w-6 h-6"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>';
        
        toast.innerHTML = `${iconHtml}<span>${message}</span>`;
        toastContainer.appendChild(toast);
        
        // Automatically remove the toast after a few seconds.
        setTimeout(() => {
            toast.classList.remove('animate-pulse');
            toast.style.opacity = '0';
            toast.style.transition = 'opacity 0.5s ease-out';
            setTimeout(() => toast.remove(), 500);
        }, 3000);
    }
    
    /**
     * @async
     * @function loadInitialChart
     * @description Fetches the initial data for the chart based on the user's
     * selected options. Resets the chart and state before loading.
     */
    async function loadInitialChart() {
        if (!sessionToken) {
            showToast('Waiting for session to start...', 'info');
            // Try to start the session again if it failed initially.
            await startSession(); 
            if(!sessionToken) return;
        }

        const startTimeStr = startTimeInput.value;
        const endTimeStr = endTimeInput.value;

        if (!startTimeStr || !endTimeStr) {
            showToast('Start Time and End Time are required.', 'error');
            return;
        }
        
        // Reset state for the new chart load.
        allDataLoaded = false;
        allChartData = [];
        allVolumeData = [];
        chartRequestId = null;
        chartCurrentOffset = 0;
        currentlyFetching = true;
        loadingIndicator.style.display = 'flex';

        const exchange = exchangeSelect.value;
        const token = symbolSelect.value;
        const interval = intervalSelect.value;

        const apiUrl = getHistoricalDataUrl(sessionToken, exchange, token, interval, startTimeStr, endTimeStr);

        try {
            const response = await fetch(apiUrl);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: response.statusText }));
                throw new Error(`HTTP error ${response.status}: ${errorData.detail || 'Failed to fetch data'}`);
            }
            
            const responseData = await response.json();

            if (!responseData || !responseData.request_id || !Array.isArray(responseData.candles) || responseData.candles.length === 0) {
                const message = responseData.message || 'No historical data available for this range.';
                showToast(message, 'info');
                candleSeries.setData([]); // Clear the chart
                volumeSeries.setData([]);
                dataSummaryElement.innerHTML = 'No data available.';
                return;
            }

            // Store the new state from the initial response.
            chartRequestId = responseData.request_id;
            chartCurrentOffset = responseData.offset;

            // Check if all data was returned in this single response.
            if (responseData.is_partial === false) {
                 allDataLoaded = true;
            }

            // Format and store the data.
            allChartData = responseData.candles.map(item => ({ time: item.unix_timestamp, open: item.open, high: item.high, low: item.low, close: item.close }));
            allVolumeData = responseData.candles.map(item => ({ time: item.unix_timestamp, value: item.volume, color: item.close > item.open ? 'rgba(16, 185, 129, 0.5)' : 'rgba(239, 68, 68, 0.5)' }));
            
            // Update the chart series with the new data.
            candleSeries.setData(allChartData);
            volumeSeries.setData(allVolumeData);
            
            // Adjust the visible range to show the most recent data.
            if (allChartData.length > 0) {
                const dataSize = allChartData.length;
                mainChart.timeScale().setVisibleLogicalRange({
                    from: Math.max(0, dataSize - 100), // Show last 100 bars or all if less than 100
                    to: dataSize - 1,
                });
            }
            
            updateDataSummary(allChartData[allChartData.length - 1], token, exchange, interval);
            showToast(responseData.message, 'success');

        } catch (error) {
            console.error('Failed to fetch initial chart data:', error);
            showToast(`Error: ${error.message}`, 'error');
            dataSummaryElement.textContent = `Error loading data: ${error.message}`;
        } finally {
            loadingIndicator.style.display = 'none';
            currentlyFetching = false;
        }
    }

    // --- Event Listeners ---
    loadChartBtn.addEventListener('click', loadInitialChart);
    themeToggle.addEventListener('click', () => {
        const newTheme = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
        applyTheme(newTheme);
    });
    window.addEventListener('resize', () => { 
        if (mainChart) mainChart.resize(chartContainer.clientWidth, chartContainer.clientHeight); 
    });

    // --- Initial Application Setup ---
    setDefaultDateTime();
    const savedTheme = localStorage.getItem('chartTheme') || 'light';
    applyTheme(savedTheme);
    if(savedTheme === 'dark') themeToggle.checked = true;

    initializeCharts();
    startSession(); // Start the session on page load.
});