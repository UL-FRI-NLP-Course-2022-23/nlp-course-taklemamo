const mongoose = require("mongoose")
const db = require("../mongodb/db")

const scorerSchema = require("../mongodb/scorer").scorerSchema

const Scorer = mongoose.model("Scorer", scorerSchema)

const addScorer = async (context, scorer) => {
    db.connectToDatabase();

    const newScorer = new Scorer(scorer)
    await newScorer.save().then((scorer) => {
        context.res = {
            status: 201,
            body: scorer,
            headers: { "Access-Control-Allow-Origin": "*" }
        };
    }).catch((err) => {
        context.res = {
            status: 400,
            body: "Error adding scorer." + err,
            headers: { "Access-Control-Allow-Origin": "*" }
        };
    });
}

const scorerExists = async (scorerId) => {
    db.connectToDatabase();

    await Scorer.find({scorerId: scorerId}).then((scorer) => {
        return scorer.length > 0;
    }).catch((err) => {
        return false;
    })    

}

module.exports = {
    scorerExists,
    addScorer
}