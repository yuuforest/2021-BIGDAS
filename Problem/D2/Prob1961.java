package Problem.D2;

import java.util.Scanner;

public class Prob1961 {
	
    public static int[][] rotate(int[][] map) {

		int M = map.length;
		int[][] map_rotate = new int[M][M];

		for (int i = 0; i < M; i++) {
			for (int j = 0; j < M; j++) map_rotate[i][j] = map[M - 1 - j][i];
		}

		return map_rotate;
	}

	public static void main(String args[]) throws Exception{
		
		Scanner sc = new Scanner(System.in);
		
		int T = sc.nextInt();

		for (int testcase = 1; testcase < T+1; testcase++) {

			int N = sc.nextInt();

			int[][] map = new int[N][N];

			for (int i = 0; i < N; i++) {
				for (int j = 0; j < N; j++) map[i][j] = sc.nextInt();
			}

			int[][] map_90 = rotate(map);
			int[][] map_180 = rotate(map_90);
			int[][] map_270 = rotate(map_180);

			System.out.println("#" + testcase);

			for (int i = 0; i < N; i++) {
				for (int j = 0; j < N; j++) System.out.print(map_90[i][j]);
				System.out.print(" ");
				for (int j = 0; j < N; j++) System.out.print(map_180[i][j]);
				System.out.print(" ");
				for (int j = 0; j < N; j++) System.out.print(map_270[i][j]);
				System.out.println();
			}
		}
		sc.close();
	}
}
