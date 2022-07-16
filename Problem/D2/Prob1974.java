package Problem.D2;

import java.util.Scanner;

public class Prob1974 {
    public static int sudoku(int[][] map) {

		int[] check1 = new int[9];
		int[] check2 = new int[9];

		for (int i = 0; i < 9; i++) {

			for (int check = 0; check < 9; check++){
				check1[check] = 0;
				check2[check] = 0;
			}

			for (int j = 0; j < 9; j++) {
				if (check1[map[i][j]-1] == 1) return 0;
				else check1[map[i][j]-1] = 1;

				if (check2[map[j][i]-1] == 1) return 0;
				else check2[map[j][i]-1] = 1;
			}
		}

		int[] check3 = new int[9];

		for (int i = 0; i < 9; i+=3) {
			for (int j = 0; j < 9; j+=3) {

				for (int check = 0; check < 9; check++) check3[check] = 0;

				for (int r = i; r < 3; r++) {
					for (int c = j; c < 3; c++) {
						if (check3[map[r][c]-1] == 1) return 0;
						else check3[map[r][c]-1] = 1;
					}
				}
			}
		}

		return 1;
	}

	public static void main(String args[]) throws Exception{
		
		Scanner sc = new Scanner(System.in);
		
		int T = sc.nextInt();

		for (int testcase = 1; testcase < T+1; testcase++) {

			int[][] map = new int[9][9];

			for (int i = 0; i < 9; i++) {
				for (int j = 0; j < 9; j++) map[i][j] = sc.nextInt();
			}

			System.out.println("#" + testcase + " " + sudoku(map));
		}
		sc.close();
	}
}
