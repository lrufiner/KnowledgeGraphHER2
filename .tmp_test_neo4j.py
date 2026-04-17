from neo4j import GraphDatabase
uri='bolt://localhost:7687'
user='neo4j'
password='password'
try:
    drv=GraphDatabase.driver(uri, auth=(user,password))
    with drv.session() as s:
        r=s.run('RETURN 1 AS v')
        v=r.single()['v']
    print('CONNECTED', v)
    drv.close()
except Exception as e:
    print('ERROR', type(e).__name__, str(e))
