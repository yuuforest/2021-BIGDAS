package Problem.D2;

import java.util.Scanner;

public class Prob1979 {

    public static int find(int[][] map, int K) {

		int len = map.length;
		int count = 0;
		int temp = 0;

		for (int i = 0; i < len; i++) {
			temp = 0;
			for (int j = 0; j < len; j++) {
				if (map[i][j] == 1) temp++;
				else {
					if (temp == K) count++;
					temp = 0;
				}
				if (j == (len-1) && temp == K) count++;
			}
		}

		for (int i = 0; i < len; i++) {
			temp = 0;
			for (int j = 0; j < len; j++) {
				if (map[j][i] == 1) temp++;
				else {
					if (temp == K) count++;
					temp = 0;
				}
				if (j == (len-1) && temp == K) count++;
			}
		}
		return count;
	}

	public static void main(String args[]) throws Exception{
		
		Scanner sc = new Scanner(System.in);
		
		int T = sc.nextInt();

		for (int testcase = 1; testcase < T+1; testcase++) {

			int N = sc.nextInt();
			int K = sc.nextInt();
			
			int[][] map = new int[N][N];
			
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < N; j++) map[i][j] = sc.nextInt();
			}

			System.out.println("#" + testcase + " " + find(map, K));
		}
		sc.close();
	}
}
