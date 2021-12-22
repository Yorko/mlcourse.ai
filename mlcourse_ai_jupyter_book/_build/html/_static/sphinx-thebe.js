/**
 * Add attributes to Thebe blocks to initialize thebe properly
 */

var initThebe = () => {
    // If Thebelab hasn't loaded, wait a bit and try again. This
    // happens because we load ClipboardJS asynchronously.
    if (window.thebelab === undefined) {
        console.log("thebe not loaded, retrying...");
        setTimeout(initThebe, 500)
        return
    }

    console.log("Adding thebe to code cells...");

    // Load thebe config in case we want to update it as some point
    thebe_config = $('script[type="text/x-thebe-config"]')[0]


    // If we already detect a Thebe cell, don't re-run
    if (document.querySelectorAll('div.thebe-cell').length > 0) {
        return;
    }

    // Update thebe buttons with loading message
    $(".thebe-launch-button").each((ii, button) => {
        button.innerHTML = `
        <div class="spinner">
            <div class="rect1"></div>
            <div class="rect2"></div>
            <div class="rect3"></div>
            <div class="rect4"></div>
        </div>
        <span class="loading-text"></span>`;
    })

    // Set thebe event hooks
    var thebeStatus;
    thebelab.on("status", function (evt, data) {
        console.log("Status changed:", data.status, data.message);

        $(".thebe-launch-button ")
        .removeClass("thebe-status-" + thebeStatus)
        .addClass("thebe-status-" + data.status)
        .find(".loading-text").html("<span class='launch_msg'>Launching from mybinder.org: </span><span class='status'>" + data.status + "</span>");

        // Now update our thebe status
        thebeStatus = data.status;

        // Find any cells with an initialization tag and ask thebe to run them when ready
        if (data.status === "ready") {
            var thebeInitCells = document.querySelectorAll('.thebe-init, .tag_thebe-init');
            thebeInitCells.forEach((cell) => {
                console.log("Initializing Thebe with cell: " + cell.id);
                cell.querySelector('.thebelab-run-button').click();
            });
        }
    });


    // Find all code cells, replace with Thebe interactive code cells
    const codeCells = document.querySelectorAll(thebe_selector)
    codeCells.forEach((codeCell, index) => {
        const codeCellId = index => `codecell${index}`;
        codeCell.id = codeCellId(index);
        codeCellText = codeCell.querySelector(thebe_selector_input);
        codeCellOutput = codeCell.querySelector(thebe_selector_output);

        // Clean up the language to make it work w/ CodeMirror and add it to the cell
        dataLanguage = detectLanguage(kernelName);

        if (codeCellText) {
            codeCellText.setAttribute('data-language', dataLanguage);
            codeCellText.setAttribute('data-executable', 'true');

            // If we had an output, insert it just after the `pre` cell
            if (codeCellOutput) {
                $(codeCellOutput).attr("data-output", "");
                $(codeCellOutput).insertAfter(codeCellText);
            }
        }
    });

    // Init thebe
    thebelab.bootstrap();
}

// Helper function to munge the language name
var detectLanguage = (language) => {
    if (language.indexOf('python') > -1) {
        language = "python";
    } else if (language === 'ir') {
        language = "r"
    }
    return language;
}
