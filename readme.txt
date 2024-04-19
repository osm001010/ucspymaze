1. 우선, 이 파일을 그대로 vscode 등에서 파일 열기로 열어야 합니다.
2. a star search를 보고 싶으시다면 AstarSearch.py 파일을 열고 실행시키면 됩니다.
3. uniform cost search를 보고 싶으시다면 uniform cost search.py를 열고 실행시키면 됩니다.
4. 현재 출력되는 것은 cost, optimal path, 그리고 탐색한 모든 점들이 print로 콘솔에 출력됩니다. 이것을 확인하기 어려우므로, 이것을 확인하고 싶으시다면 각각 파일에서 manager.show_solution_animation(maze.id) 부분을 주석 처리하시면 됩니다.
5. solution animation의 경우, 지나가는 모든 점이 아니라, 새로 탐색하게 되는 점들만을 출력하게 됩니다. 이를 통해 직관성을 확보했습니다.