# Proteus Docker 실행 안내

한국어 기본 실행 순서는 저장소 루트의 [docker.md](../../docker.md)에 정리돼 있습니다.

- GPU·모델·SLO 설정
- Docker 이미지 빌드와 사전 학습 가중치 다운로드
- 워커 실행과 실제 GPU 프로파일링
- gRPC 요청과 첫 요청·반복 요청 테스트
- MILP·적응형 배치 확인과 부하 테스트
- 로그 확인과 종료 및 결과 저장 위치

설정 파일은 [proteus-docker.env](proteus-docker.env), 실행 스크립트는
[proteus-docker.sh](proteus-docker.sh), 모델 목록은 [models.json](models.json)입니다.
정책 구현 범위와 상세 실험 기준은 [GPU_SERVING.md](../../GPU_SERVING.md)를 참고하세요.
