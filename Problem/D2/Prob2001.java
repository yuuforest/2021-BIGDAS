package Problem.D2;

import java.util.Scanner;

public class Prob2001 {
    public static int fly(int[][] map, int M, int row, int column) {

		int sum = 0;

		for (int i = 0; i < M; i++) {
			for (int j = 0; j < M; j++) {
				sum += map[row+i][column+j];
			}
		}

		return sum;
	}

	public static void main(String args[]) throws Exception{
		
		Scanner sc = new Scanner(System.in);
		
		int T = sc.nextInt();

		for (int testcase = 1; testcase < T+1; testcase++) {

			int N = sc.nextInt();
			int M = sc.nextInt();
			
			int[][] map = new int[N][N];
			
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < N; j++) map[i][j] = sc.nextInt();
			}

			int max = 0;

			for (int i = 0; i < N-M+1; i++) {
				for (int j = 0; j < N-M+1; j++) {
					max = Math.max(max, fly(map, M, i, j));
				}			
			}

			System.out.println("#" + testcase + " " + max);
		}
		sc.close();
	}
}
