from elasticsearch import Elasticsearch as ES


class ESClient(BaseException):

    def __init__(self):
        self.es = self.createESClient()
        self.es.indices.create(index = 'megacorp', ignore = 400)

    def createESClient(self) -> object:
        es = ES(["http://localhost:9200"], timeout=3600)
        return es

    def insertES(self, index, body, id) -> bool:
        return self.es.index(index = index, body = body, id = id)

    def queryES(self, index, body):
        return self.es.search(index = index, body = body)

    def updateES(self, index, body):
        return self.es.update(index = index, body = body, id = 100)

    def deleteES(self, index, id):
        return self.es.delete(index = index, id = id)



if __name__ == '__main__':
    esClient = ESClient()
    body = {"first_name":"xiao","last_name":"xiao", 'age': 25, 'about': 'I love to go rock climbing', 'interests': ['game', 'play']}
    doc_type = 'doc'
    id = 100
    query = {
        "query": {
            "match_all": {}
        }
    }
    index = "megacorp"
    esClient.insertES(index, body, 100)
    result = esClient.queryES(index, query)
    print(result)

    updateBody = {"doc":{"first_name":"hong","last_name":"xiao", 'age': 25, 'about': 'I love to go rock climbing', 'interests': ['game', 'play']}}
    result = esClient.updateES(index, str(updateBody))
    print(result)
