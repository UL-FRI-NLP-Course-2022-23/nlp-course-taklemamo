const data = {
    todos: [
      {
        id: 1,
        task: 'task1',
        assignee: 'assignee1000',
        status: 'completed'
      },
      {
        id: 2,
        task: 'task2',
        assignee: 'assignee1001',
        status: 'completed'
      },
      {
        id: 3,
        task: 'task3',
        assignee: 'assignee1002',
        status: 'completed'
      },
      {
        id: 4,
        task: 'task4',
        assignee: 'assignee1000',
        status: 'completed'
      },
      
    ]
  };

  module.exports = {
    getToDos: function() {
        return data.todos;
    },
    addToDo: function(task) {
        task.id = data.todos.length + 1;
        data.todos.push(task);
        return {
            message: "task added",
            tasks: data.todos.length
        }
    }
  }