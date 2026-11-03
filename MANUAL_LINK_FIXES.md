# 성호님 직접 수정할 링크 위치

기준: INF-25 통합 작업 브랜치의 README.ko.md. 이 표에 해당하는 학생 링크·명령어는 수정하지 않고 보존했습니다. 링크를 고친 뒤 링크 검증을 다시 실행해야 합니다.

| 파일·줄 | 직접 수정할 내용 | 올바른 대상 |
|---|---|---|
| [README.ko.md:39](README.ko.md#L39) | 설치 명령어의 Markdown 링크 제거 | `git clone https://github.com/humblebeeai/infer-ci.git` |
| [README.ko.md:111](README.ko.md#L111) | 잘못된 외부 주소를 저장소 상대 경로로 변경 | `docs/object-detection-metrics-usage.md` |
| [README.ko.md:180](README.ko.md#L180) | 잘못된 외부 주소를 저장소 상대 경로로 변경 | `docs/dev/scripts/clean.sh.md` |
| [README.ko.md:181](README.ko.md#L181) | 잘못된 외부 주소를 저장소 상대 경로로 변경 | `docs/dev/scripts/get-version.sh.md` |
| [README.ko.md:182](README.ko.md#L182) | 잘못된 외부 주소를 저장소 상대 경로로 변경 | `docs/dev/scripts/test.sh.md` |
| [README.ko.md:183](README.ko.md#L183) | 잘못된 외부 주소를 저장소 상대 경로로 변경 | `docs/dev/scripts/bump-version.sh.md` |
| [README.ko.md:184](README.ko.md#L184) | 잘못된 외부 주소를 저장소 상대 경로로 변경 | `docs/dev/scripts/build.sh.md` |
| [README.ko.md:185](README.ko.md#L185) | 잘못된 외부 주소를 저장소 상대 경로로 변경 | `docs/dev/scripts/release.sh.md` |
| [README.ko.md:186](README.ko.md#L186) | 잘못된 외부 주소를 저장소 상대 경로로 변경 | `docs/dev/scripts/changelog.sh.md` |
| [README.ko.md:187](README.ko.md#L187) | 잘못된 외부 주소를 저장소 상대 경로로 변경 | `docs/dev/scripts/diagrams.sh.md` |
| [README.ko.md:188](README.ko.md#L188) | 잘못된 외부 주소를 저장소 상대 경로로 변경 | `docs/dev/scripts/docs.sh.md` |

문서 링크는 `[표시할 이름](위 상대 경로)` 형태로 변경합니다. 설치 명령어는 명령만 남깁니다.

별도 확인: INF-24 Linear 댓글의 이전 개인 저장소 PR #1 링크는 PR #40은 병합 없이 닫혔으므로, 최신 검토 대상인 공식 PR #39와 역할별 변경 내용을 기록해야 합니다. 이 댓글은 수정하지 않았습니다.

## 수정 후 확인

문서 도구 의존성을 설치한 환경에서 아래 명령을 실행합니다. 수정 전에는 11개 미해결 항목으로 종료하는 것이 현재 상태와 일치합니다.

```sh
python scripts/check-korean-readme.py
```

이 점검은 로컬 파일 대상, 잘못된 문서 URL, clone 명령의 Markdown 문법, 영어·한국어 Python 예제 일치를 확인합니다. 외부 HTTP 주소의 실제 응답은 확인하지 않습니다.
