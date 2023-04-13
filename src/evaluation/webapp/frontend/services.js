async function getText(id) {
    /*
    const textSchema = new mongoose.Schema({
        textId: String,
        originalText: String,
        paraphrasedText: String
    });
    */
    const response = await fetch(
        "https://nlpprojtest.azurewebsites.net/api/texts/" + id +
        "?code=csyFDq1DGzghi5_H36oOUtda68CVDrE6WBtYphM6uLHLAzFu1sEVAw%3D%3D", 
        {
            method: "GET"
        })
    const data = await response.json()
    return data[0]
}


async function postScore(score) {
    /*
    mongoose.Schema({
        textId: String,
        scorerId: String,
        appropriateness: Number,
        fluency: Number,
        diversity: Number,
        overall: Number
    });
    */
    const response = await fetch(
        "https://nlpprojtest.azurewebsites.net/api/scores?code=NfYQxghsyNFDDmrS4YDSgFedAurPYryvKNj8eXX0f3AmAzFuQUlfog%3D%3D", 
        //"http://localhost:7071/api/scores",
        {
            method: "POST",
            headers: {
                'Content-Type': 'text/plain',
            },
            body: JSON.stringify(score)
        })
    const data = await response.json()
    return data
}