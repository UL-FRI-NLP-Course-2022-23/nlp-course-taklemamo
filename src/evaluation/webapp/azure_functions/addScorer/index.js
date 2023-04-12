const scorerService = require('../services/scorerService');

module.exports = async function (context, req) {
    context.log('JavaScript HTTP trigger function processed a request.');

    await scorerService.addScorer(context, req.body);
}