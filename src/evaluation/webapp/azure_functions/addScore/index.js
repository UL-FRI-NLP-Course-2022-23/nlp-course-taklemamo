const scoreService = require('../services/scoreService');

module.exports = async function (context, req) {
    context.log('JavaScript HTTP trigger function processed a request.');

    if (req.body) {
        await scoreService.addScore(context, req.body);
    }
    else {
        context.res = {
            status: 400,
            body: "Score not added. No body in request."
        };
    }
}