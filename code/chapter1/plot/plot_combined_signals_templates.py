# Plotly 点击面板脚本和 dashboard HTML 模板。
CLICK_INFO_SCRIPT = """
(function() {
    var graph = document.getElementById('{plot_id}');
    if (!graph) {
        return;
    }

    document.documentElement.style.height = '100%';
    document.documentElement.style.margin = '0';
    document.body.style.height = '100vh';
    document.body.style.margin = '0';
    document.body.style.padding = '10px 12px';
    document.body.style.boxSizing = 'border-box';
    document.body.style.overflow = 'hidden';

    var panel = document.createElement('div');
    panel.style.margin = '0 0 8px 0';
    panel.style.padding = '10px 12px';
    panel.style.border = '1px solid #d1d5db';
    panel.style.borderRadius = '6px';
    panel.style.background = '#f9fafb';
    panel.style.color = '#111827';
    panel.style.font = '14px Arial, sans-serif';
    panel.textContent = 'Click a signal point to show its date and factor.';
    graph.parentNode.insertBefore(panel, graph);

    function resizeGraph() {
        var bodyStyle = window.getComputedStyle(document.body);
        var verticalPadding = (
            parseFloat(bodyStyle.paddingTop) +
            parseFloat(bodyStyle.paddingBottom)
        ) || 0;
        var horizontalPadding = (
            parseFloat(bodyStyle.paddingLeft) +
            parseFloat(bodyStyle.paddingRight)
        ) || 0;
        var panelHeight = panel.getBoundingClientRect().height;
        var availableHeight = window.innerHeight - verticalPadding - panelHeight - 8;
        var availableWidth = document.body.clientWidth - horizontalPadding;

        availableHeight = Math.max(120, Math.floor(availableHeight));
        availableWidth = Math.max(260, Math.floor(availableWidth));
        graph.style.width = availableWidth + 'px';
        graph.style.height = availableHeight + 'px';

        Plotly.relayout(graph, {
            autosize: true,
            width: availableWidth,
            height: availableHeight
        });
    }

    window.addEventListener('resize', resizeGraph);
    window.requestAnimationFrame(resizeGraph);

    graph.on('plotly_click', function(eventData) {
        if (!eventData || !eventData.points || eventData.points.length === 0) {
            return;
        }

        var point = eventData.points[0];
        if (!point.customdata) {
            return;
        }

        var dateText = point.customdata[0];
        var factorLabel = point.customdata[1];
        var factorName = point.customdata[2];
        var factorValue = point.customdata[4] || '';
        var closeText = Number(point.y).toLocaleString(
            undefined,
            {maximumFractionDigits: 4}
        );

        panel.innerHTML = (
            '<strong>Date:</strong> ' + dateText +
            '&nbsp;&nbsp; <strong>Factor:</strong> ' + factorLabel +
            '&nbsp;&nbsp; <strong>Name:</strong> ' + factorName +
            '&nbsp;&nbsp; <strong>Close:</strong> ' + closeText +
            (factorValue ? '&nbsp;&nbsp; <strong>Factor value:</strong> ' + factorValue : '')
        );
        resizeGraph();

        Plotly.relayout(graph, {
            annotations: [{
                x: point.x,
                y: point.y,
                xref: 'x',
                yref: 'y',
                text: dateText + '<br>' + factorLabel,
                showarrow: true,
                arrowhead: 2,
                ax: 32,
                ay: -46,
                bgcolor: 'rgba(255,255,255,0.94)',
                bordercolor: point.fullData.marker.color || '#111827',
                borderwidth: 1,
                font: {size: 12, color: '#111827'}
            }]
        });
    });
})();
"""
DASHBOARD_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Combined Factor Signals Dashboard</title>
    <script src="plotly.min.js"></script>
    <style>
        html,
        body {{
            height: 100%;
            width: 100%;
            margin: 0;
            overflow: hidden;
            background: #f7f8fb;
            color: #111827;
            font-family: Arial, sans-serif;
        }}

        body {{
            overflow: hidden;
        }}

        .shell {{
            display: grid;
            grid-template-columns: minmax(220px, 280px) 1fr;
            position: fixed;
            inset: 0;
            width: 100vw;
            height: 100vh;
            height: 100dvh;
            min-height: 0;
            overflow: hidden;
        }}

        .sidebar {{
            display: flex;
            flex-direction: column;
            height: 100%;
            min-width: 0;
            min-height: 0;
            overflow: hidden;
            border-right: 1px solid #d9dee8;
            background: #ffffff;
        }}

        .sidebar-header {{
            flex: 0 0 auto;
            padding: 14px 14px 10px;
            border-bottom: 1px solid #e5e7eb;
        }}

        .sidebar-title {{
            font-size: 16px;
            font-weight: 700;
            line-height: 1.2;
        }}

        .sidebar-meta {{
            margin-top: 4px;
            color: #667085;
            font-size: 12px;
        }}

        .search {{
            width: 100%;
            box-sizing: border-box;
            margin-top: 12px;
            padding: 8px 9px;
            border: 1px solid #cfd5df;
            border-radius: 6px;
            color: #111827;
            background: #ffffff;
            font-size: 13px;
            outline: none;
        }}

        .search:focus {{
            border-color: #2563eb;
            box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.12);
        }}

        .symbol-list {{
            flex: 1 1 auto;
            height: 0;
            min-height: 0;
            overflow-x: hidden;
            overflow-y: auto;
            overscroll-behavior: contain;
            padding: 8px;
        }}

        .symbol-button {{
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 8px;
            align-items: center;
            width: 100%;
            min-height: 36px;
            margin: 0 0 4px;
            padding: 8px 9px;
            border: 1px solid transparent;
            border-radius: 6px;
            background: transparent;
            color: #111827;
            cursor: pointer;
            font: inherit;
            text-align: left;
        }}

        .symbol-button:hover {{
            background: #f2f5f9;
        }}

        .symbol-button.active {{
            border-color: #9bb7f0;
            background: #eaf1ff;
        }}

        .symbol-name {{
            min-width: 0;
            overflow: hidden;
            font-weight: 700;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .signal-count {{
            color: #667085;
            font-size: 12px;
            white-space: nowrap;
        }}

        .content {{
            display: flex;
            flex-direction: column;
            height: 100%;
            min-width: 0;
            min-height: 0;
            overflow: hidden;
        }}

        .topbar {{
            display: grid;
            flex: 0 0 auto;
            grid-template-columns: 1fr auto;
            gap: 16px;
            align-items: center;
            min-height: 44px;
            padding: 8px 14px;
            border-bottom: 1px solid #d9dee8;
            background: #ffffff;
        }}

        .title {{
            min-width: 0;
            overflow: hidden;
            font-size: 17px;
            font-weight: 700;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .summary {{
            color: #667085;
            font-size: 13px;
            white-space: nowrap;
        }}

        .info {{
            flex: 0 0 auto;
            min-height: 18px;
            padding: 6px 14px;
            border-bottom: 1px solid #e5e7eb;
            background: #fbfcfe;
            color: #344054;
            font-size: 13px;
        }}

        .plot-wrap {{
            flex: 1;
            flex-basis: 0;
            min-height: 0;
            max-height: 100%;
            padding: 0;
            overflow: hidden;
            position: relative;
        }}

        #dashboard-plot {{
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
        }}

        @media (max-width: 760px) {{
            body {{
                overflow: hidden;
            }}

            .shell {{
                grid-template-columns: 1fr;
                grid-template-rows: minmax(150px, 32vh) 1fr;
                height: 100vh;
                height: 100dvh;
                min-height: 0;
                overflow: hidden;
            }}

            .sidebar {{
                height: 100%;
                min-height: 0;
                overflow: hidden;
                border-right: 0;
                border-bottom: 1px solid #d9dee8;
            }}

            .topbar {{
                grid-template-columns: 1fr;
                gap: 4px;
            }}

            .summary {{
                white-space: normal;
            }}
        }}
    </style>
</head>
<body>
    <div class="shell">
        <aside class="sidebar">
            <div class="sidebar-header">
                <div class="sidebar-title">Symbols</div>
                <div class="sidebar-meta" id="symbol-total"></div>
                <input class="search" id="symbol-search" placeholder="Filter symbols">
            </div>
            <nav class="symbol-list" id="symbol-list"></nav>
        </aside>
        <main class="content">
            <header class="topbar">
                <div class="title" id="current-title"></div>
                <div class="summary" id="current-summary"></div>
            </header>
            <div class="info" id="point-info">Click a signal point to show its date and factor.</div>
            <section class="plot-wrap">
                <div id="dashboard-plot"></div>
            </section>
        </main>
    </div>
    <script>
        const dashboardData = {dashboard_data};
        const symbolOrder = dashboardData.symbols;
        const symbolData = dashboardData.data;
        let currentSymbol = symbolOrder[0] || null;

        const graph = document.getElementById("dashboard-plot");
        const plotWrap = graph.parentElement;
        const content = document.querySelector(".content");
        const list = document.getElementById("symbol-list");
        const search = document.getElementById("symbol-search");
        const total = document.getElementById("symbol-total");
        const title = document.getElementById("current-title");
        const summary = document.getElementById("current-summary");
        const info = document.getElementById("point-info");
        let clickHandlerBound = false;

        total.textContent = symbolOrder.length + " symbols";

        function formatClose(value) {{
            if (value === null || value === undefined || Number.isNaN(Number(value))) {{
                return "";
            }}
            return Number(value).toLocaleString(undefined, {{
                maximumFractionDigits: 4
            }});
        }}

        function renderSymbolList() {{
            const filter = search.value.trim().toUpperCase();
            list.innerHTML = "";

            symbolOrder.forEach(function(symbol) {{
                if (filter && !symbol.includes(filter)) {{
                    return;
                }}

                const data = symbolData[symbol];
                const button = document.createElement("button");
                button.type = "button";
                button.className = "symbol-button" + (
                    symbol === currentSymbol ? " active" : ""
                );
                button.dataset.symbol = symbol;
                button.innerHTML = (
                    '<span class="symbol-name">' + symbol + '</span>' +
                    '<span class="signal-count">' + data.signal_count + ' signals</span>'
                );
                button.addEventListener("click", function() {{
                    selectSymbol(symbol);
                }});
                list.appendChild(button);
            }});
        }}

        function buildTraces(data) {{
            const traces = [{{
                x: data.price.dates,
                y: data.price.close,
                mode: "lines",
                name: "close",
                line: {{
                    color: "#111827",
                    width: 1.5
                }},
                hovertemplate: (
                    "date=%{{x}}<br>" +
                    "close=%{{y:.4f}}<extra>close</extra>"
                )
            }}];

            data.factors.forEach(function(factor) {{
                if (!factor.dates.length) {{
                    return;
                }}

                traces.push({{
                    x: factor.dates,
                    y: factor.close,
                    mode: "markers",
                    name: factor.label + " (" + factor.dates.length + ")",
                    customdata: factor.customdata,
                    marker: {{
                        symbol: factor.marker,
                        size: 11,
                        color: factor.color,
                        line: {{
                            width: 1.8,
                            color: factor.color
                        }}
                    }},
                    hovertemplate: (
                        "date=%{{customdata[0]}}<br>" +
                        "factor=%{{customdata[1]}}<br>" +
                        "name=%{{customdata[2]}}<br>" +
                        "close=%{{y:.4f}}<br>" +
                        "factor value=%{{customdata[4]}}<br>" +
                        "<extra></extra>"
                    )
                }});
            }});

            return traces;
        }}

        function buildLayout(data) {{
            return {{
                title: {{
                    text: ""
                }},
                autosize: true,
                template: "plotly_white",
                clickmode: "event+select",
                hovermode: "closest",
                dragmode: "zoom",
                xaxis: {{
                    title: "Date",
                    showgrid: true,
                    gridcolor: "rgba(17, 24, 39, 0.08)",
                    rangeslider: {{
                        visible: true,
                        thickness: 0.045
                    }}
                }},
                yaxis: {{
                    title: "Close price",
                    showgrid: true,
                    gridcolor: "rgba(17, 24, 39, 0.08)"
                }},
                legend: {{
                    orientation: "v",
                    x: 1.02,
                    xanchor: "left",
                    y: 1,
                    yanchor: "top"
                }},
                margin: {{
                    l: 58,
                    r: 210,
                    t: 8,
                    b: 42
                }}
            }};
        }}

        function dashboardPlotSize() {{
            const viewportHeight = (
                window.visualViewport && window.visualViewport.height
            ) ? window.visualViewport.height : window.innerHeight;
            const viewportWidth = (
                window.visualViewport && window.visualViewport.width
            ) ? window.visualViewport.width : window.innerWidth;
            const contentRect = content.getBoundingClientRect();
            const rect = plotWrap.getBoundingClientRect();
            const rightEdge = Math.min(contentRect.right, viewportWidth);
            const bottomEdge = Math.min(contentRect.bottom, viewportHeight);
            const safeBottom = 12;
            const availableHeight = Math.max(
                160,
                Math.floor(bottomEdge - rect.top - safeBottom)
            );
            const availableWidth = Math.max(
                320,
                Math.floor(rightEdge - rect.left)
            );

            plotWrap.style.height = availableHeight + "px";
            plotWrap.style.maxHeight = availableHeight + "px";

            return {{
                width: availableWidth,
                height: availableHeight
            }};
        }}

        function resizeDashboardPlot() {{
            if (!graph.data) {{
                return;
            }}

            const size = dashboardPlotSize();
            graph.style.width = size.width + "px";
            graph.style.height = size.height + "px";
            Plotly.relayout(graph, {{
                autosize: true,
                width: size.width,
                height: size.height
            }});
        }}

        function updateHeader(data) {{
            title.textContent = data.symbol;
            summary.textContent = (
                data.price.dates.length + " daily points, " +
                data.factor_count + " factors, " +
                data.signal_count + " signals"
            );
            info.textContent = "Click a signal point to show its date and factor.";
        }}

        function selectSymbol(symbol) {{
            currentSymbol = symbol;
            const data = symbolData[symbol];
            renderSymbolList();
            updateHeader(data);

            Plotly.react(
                graph,
                buildTraces(data),
                {{
                    ...buildLayout(data),
                    ...dashboardPlotSize()
                }},
                {{
                    displaylogo: false,
                    responsive: true,
                    scrollZoom: true
                }}
            ).then(function() {{
                bindPlotClick();
                resizeDashboardPlot();
            }});
        }}

        function bindPlotClick() {{
            if (clickHandlerBound || typeof graph.on !== "function") {{
                return;
            }}

            graph.on("plotly_click", handlePlotClick);
            clickHandlerBound = true;
        }}

        function handlePlotClick(eventData) {{
            if (!eventData || !eventData.points || eventData.points.length === 0) {{
                return;
            }}

            const point = eventData.points[0];
            if (!point.customdata) {{
                return;
            }}

            const dateText = point.customdata[0];
            const factorLabel = point.customdata[1];
            const factorName = point.customdata[2];
            const factorValue = point.customdata[4] || "";
            const closeText = formatClose(point.y);

            info.innerHTML = (
                "<strong>Date:</strong> " + dateText +
                "&nbsp;&nbsp; <strong>Factor:</strong> " + factorLabel +
                "&nbsp;&nbsp; <strong>Name:</strong> " + factorName +
                "&nbsp;&nbsp; <strong>Close:</strong> " + closeText +
                (factorValue ? "&nbsp;&nbsp; <strong>Factor value:</strong> " + factorValue : "")
            );

            Plotly.relayout(graph, {{
                annotations: [{{
                    x: point.x,
                    y: point.y,
                    xref: "x",
                    yref: "y",
                    text: dateText + "<br>" + factorLabel,
                    showarrow: true,
                    arrowhead: 2,
                    ax: 32,
                    ay: -46,
                    bgcolor: "rgba(255,255,255,0.94)",
                    bordercolor: point.fullData.marker.color || "#111827",
                    borderwidth: 1,
                    font: {{
                        size: 12,
                        color: "#111827"
                    }}
                }}]
            }});
        }}

        search.addEventListener("input", renderSymbolList);
        window.addEventListener("resize", function() {{
            resizeDashboardPlot();
        }});
        if (window.visualViewport) {{
            window.visualViewport.addEventListener("resize", resizeDashboardPlot);
            window.visualViewport.addEventListener("scroll", resizeDashboardPlot);
        }}

        renderSymbolList();
        if (currentSymbol) {{
            selectSymbol(currentSymbol);
        }}
    </script>
</body>
</html>
"""
