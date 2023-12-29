


class edgeNode:
    def __init__(self, studentId, questionId, conceptId, timestamp, value_memory_matrix):
        self.studentId = studentId
        self.questionId = questionId
        self.conceptId = conceptId
        self.timestamp = timestamp
        self.value_memory_matrix = value_memory_matrix

    def getStudentId(self):
        return self.studentId

    def getQuestionId(self):
        return self.questionId

    def getConceptId(self):
        return self.conceptId

    def getTimestamp(self):
        return self.timestamp

    def getValueMemoryMatrix(self):
        return self.value_memory_matrix


class questionNode(edgeNode):

    def __init__(self, questionId, conceptId, timestamp, value_memory_matrix):
        super(questionNode, self).__init__(-1, questionId, conceptId, timestamp, value_memory_matrix)
        self.related_student_node = list()

    def addStudentNode(self, edgeNode: edgeNode):
        curIndex = len(self.related_question_node)
        self.related_student_node.insert(curIndex, edgeNode)

class studentNode(edgeNode):

    def __init__(self, studentId):
        super(studentNode, self).__init__(studentId, -1, -1, -1, -1)
        self.related_question_node = list()

    def addStudentNode(self, edgeNode: edgeNode):
        curIndex = len(self.related_question_node)
        self.related_question_node.insert(curIndex, edgeNode)

    def printCurrelatedNode(self):
        for node in self.related_question_node:
            print(node)


if __name__ == '__main__':
    print("00")
    pp = studentNode(5)
    question1 = questionNode(5,5,5,5)
    pp.addStudentNode(question1)
    pp.printCurrelatedNode()
