const data = require('./data.js');

module.exports = {
    getTodos: function(context) {
        try {
            const vacations = data.getToDos();
            context.res.status(200).json(vacations);
          } catch (error) {
            context.res.status(500).send(error);
        }
    },
    addTodos: function(context) {
        try {
            const response = data.addToDo(context.req.body);
            context.res.status(200).json(response);
          } catch (error) {
            context.res.status(500).send(error);
        }
    }
}