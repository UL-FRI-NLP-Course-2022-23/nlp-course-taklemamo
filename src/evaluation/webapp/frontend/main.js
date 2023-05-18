let i = -1;

async function setText() {
    // next text is id + num
    const text = await getText("naduskladitev-" + i)
    document.getElementById("text_original").innerHTML = text.originalText
    document.getElementById("text_par").innerHTML = text.paraphrasedText
    document.getElementById("text_id").innerHTML = text.textId
}

window.onload = async function() {
    // save consecutive text id num in local storage
    if (localStorage.getItem("id_num") != null) {
        i = localStorage.getItem("id_num")
    }
    else {
        i = 0
        localStorage.setItem("id_num", i)
    }

    await setText()

    let form = document.getElementById("text_form");
    form.addEventListener('submit', handleForm);
}

async function handleForm(e) { 
    e.preventDefault(); 

    console.log("submitting score")

    let text_id = $("#text_id").text()
    let scorer_id = $("#scorer_id").val()
    let appropriateness = $("#range_apr").val()
    let fluency = $("#range_flue").val()
    let diversity = $("#range_div").val()
    let overall = (parseFloat(appropriateness) + parseFloat(fluency) + parseFloat(diversity)) / 3
    let score = {
        textId: text_id,
        scorerId: scorer_id,
        appropriateness: appropriateness,
        fluency: fluency,
        diversity: diversity,
        overall: overall
    }

    // save score to db
    let status = await postScore(score)
    console.log(status)

    // next text
    i++
    localStorage.setItem("id_num", i)

    await setText()
} 
