const scoreService = require('../services/scoreService');

module.exports = async function (context, req) {
    context.log('JavaScript HTTP trigger function processed a request.');

    await scoreService.getScores(context);
}