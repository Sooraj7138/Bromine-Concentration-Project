def register_water_level_routes(app):
    from flask import render_template, request, redirect, url_for, send_file, flash
    import os
    from Final_concentration_service import forecast_future_C2

    UPLOAD_FOLDER = os.path.join(app.config['UPLOAD_FOLDER'], 'evaporation')
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # ensures folder exists
    app.config['UPLOAD_FOLDER_EVAP'] = UPLOAD_FOLDER
    ALLOWED_EXTENSIONS = {'csv'}

    # === Helper ===
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @app.route('/Water-Level', methods=['GET','POST'])
    def Waterlevel():
        return render_template('Water-Level.html')

    @app.route('/Water-Level-Predict', methods=['GET', 'POST'])
    def Waterlevelpredict():
        if request.method == 'POST':
            user_start = request.form.get("startMonthEvap")  # Format: YYYY-MM
            user_end = request.form.get("endMonthEvap")
            try:
                # Get toggle value: 'a' for Zone, 'b' for Pond
                mode = request.form.get("toggle", "a")
                selected_zone = request.form.get("zone_select") if mode == 'a' else None
                selected_pond = request.form.get("pond_select") if mode == 'b' else None

                prediction_df, plot_html = forecast_future_C2(user_start, user_end)

                # Optional cleanup of columns before saving
                columns_to_drop = ['Day', 'Month', 'Year', 'Month_sin', 'Month_cos']
                prediction_df = prediction_df.drop(columns=columns_to_drop, errors='ignore')
                prediction_df.to_csv(
                    "D:/Sooraj/Project_Bromine_Concentration/app/outputs/trainer_prediction_output.csv", index=False)

                # Build the output label
                if mode == 'a':
                    prediction_label = "Prediction for Zone-B"
                elif mode == 'b' and selected_pond:
                    prediction_label = f"Prediction for {selected_pond}"
                else:
                    prediction_label = "Prediction"

                # Table styling
                style = """
                <style>
                    table {
                        width: 100%;
                        border-collapse: collapse;
                        text-align: center;
                        color: white;
                    }
                    th, td {
                        text-align: center;
                        vertical-align: middle;
                    }
                </style>
                """
                # table_html = style + prediction_df.to_html(classes="table table-striped table-bordered",
                #                                            index=False)

                # Convert DataFrame to JSON for AG Grid
                table_data = prediction_df.to_dict(orient='records')  # list of dicts
                columns = [{"headerName": col, "field": col} for col in prediction_df.columns]

                return render_template(
                    'Water-Level.html',
                    # table_html=table_html,
                    table_data=table_data,
                    columns=columns,
                    plot_html=plot_html,
                    prediction_label=prediction_label,
                    show_output=True
                )

            except Exception as e:
                flash(f"Prediction Failed due to {str(e)} error. Please check the file format!", "error")
                print(f"Prediction failed: {str(e)}")
                return redirect(url_for('Waterlevel'))

        else:
            flash("Invalid File Type. Please upload a CSV file!", "error")
            print('Invalid file type')
            return redirect(url_for('Waterlevel'))

        return render_template('Water-Level.html')


    @app.route("/download_predictions_water-level")
    def download_predictions_Water():
        try:
            file_path = "D:/Sooraj/Project_Bromine_Concentration/app/outputs/trainer_prediction_output.csv"

            if not os.path.exists(file_path):
                flash("❌ No predictions available to download.")
                return redirect(url_for("Waterlevelpredict"))

            return send_file(
                file_path,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Water_level_Prediction.csv"
            )

        except Exception as e:
            print("Download error:", e)
            flash("❌ Failed to download predictions.")
            return redirect(url_for("Waterlevelpredict"))