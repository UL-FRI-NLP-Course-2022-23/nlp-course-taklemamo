const mongoose = require("mongoose")
const db = require("../mongodb/db")
const scorerService = require("./scorerService")

const scoreSchema = require("../mongodb/score").scoreSchema
const Score = mongoose.model("Score", scoreSchema)

const getScores = async (context) => {
    db.connectToDatabase();

    await Score.find().then((scores) => {
        context.res = {
            status: 200,
            body: scores,
            headers: { "Access-Control-Allow-Origin": "*" }
        };
    }).catch((err) => { 
        context.res = {
            status: 400,
            body: "Error getting scores." + err,
            headers: { "Access-Control-Allow-Origin": "*" }
        };
    });
}

const addScore = async (context, score) => {
    const scorerExists = await scorerService.scorerExists(score.scorerId);

    if (!scorerExists) {
        context.res = {
            status: 400,
            body: "Error adding score. Scorer does not exist.",
            headers: { "Access-Control-Allow-Origin": "*" }
        };
        return;
    }

    db.connectToDatabase();

    const newScore = new Score(score)
    await newScore.save().then((score) => {
        context.res = {
            status: 201,
            body: score,
            headers: { "Access-Control-Allow-Origin": "*" }
        };
    }).catch((err) => {
        context.res = {
            status: 400,
            body: "Error adding score." + err,
            headers: { "Access-Control-Allow-Origin": "*" }
        };
    });
}

module.exports = {
    getScores,
    addScore
}