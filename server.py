from openreward.environments import Server

from swe_atlas_qna import SWEAtlasQnA

if __name__ == "__main__":
    server = Server([SWEAtlasQnA])
    server.run()
