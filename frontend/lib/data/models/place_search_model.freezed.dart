// GENERATED CODE - DO NOT MODIFY BY HAND
// coverage:ignore-file
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'place_search_model.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

// dart format off
T _$identity<T>(T value) => value;
PlaceSearchModel _$PlaceSearchModelFromJson(
  Map<String, dynamic> json
) {
    return _PlaceSearchResponse.fromJson(
      json
    );
}

/// @nodoc
mixin _$PlaceSearchModel {

@JsonKey(name: 'documents') List<LocationModel> get places;
/// Create a copy of PlaceSearchModel
/// with the given fields replaced by the non-null parameter values.
@JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
$PlaceSearchModelCopyWith<PlaceSearchModel> get copyWith => _$PlaceSearchModelCopyWithImpl<PlaceSearchModel>(this as PlaceSearchModel, _$identity);

  /// Serializes this PlaceSearchModel to a JSON map.
  Map<String, dynamic> toJson();


@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is PlaceSearchModel&&const DeepCollectionEquality().equals(other.places, places));
}

@JsonKey(includeFromJson: false, includeToJson: false)
@override
int get hashCode => Object.hash(runtimeType,const DeepCollectionEquality().hash(places));

@override
String toString() {
  return 'PlaceSearchModel(places: $places)';
}


}

/// @nodoc
abstract mixin class $PlaceSearchModelCopyWith<$Res>  {
  factory $PlaceSearchModelCopyWith(PlaceSearchModel value, $Res Function(PlaceSearchModel) _then) = _$PlaceSearchModelCopyWithImpl;
@useResult
$Res call({
@JsonKey(name: 'documents') List<LocationModel> places
});




}
/// @nodoc
class _$PlaceSearchModelCopyWithImpl<$Res>
    implements $PlaceSearchModelCopyWith<$Res> {
  _$PlaceSearchModelCopyWithImpl(this._self, this._then);

  final PlaceSearchModel _self;
  final $Res Function(PlaceSearchModel) _then;

/// Create a copy of PlaceSearchModel
/// with the given fields replaced by the non-null parameter values.
@pragma('vm:prefer-inline') @override $Res call({Object? places = null,}) {
  return _then(_self.copyWith(
places: null == places ? _self.places : places // ignore: cast_nullable_to_non_nullable
as List<LocationModel>,
  ));
}

}


/// Adds pattern-matching-related methods to [PlaceSearchModel].
extension PlaceSearchModelPatterns on PlaceSearchModel {
/// A variant of `map` that fallback to returning `orElse`.
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case final Subclass value:
///     return ...;
///   case _:
///     return orElse();
/// }
/// ```

@optionalTypeArgs TResult maybeMap<TResult extends Object?>(TResult Function( _PlaceSearchResponse value)?  $default,{required TResult orElse(),}){
final _that = this;
switch (_that) {
case _PlaceSearchResponse() when $default != null:
return $default(_that);case _:
  return orElse();

}
}
/// A `switch`-like method, using callbacks.
///
/// Callbacks receives the raw object, upcasted.
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case final Subclass value:
///     return ...;
///   case final Subclass2 value:
///     return ...;
/// }
/// ```

@optionalTypeArgs TResult map<TResult extends Object?>(TResult Function( _PlaceSearchResponse value)  $default,){
final _that = this;
switch (_that) {
case _PlaceSearchResponse():
return $default(_that);case _:
  throw StateError('Unexpected subclass');

}
}
/// A variant of `map` that fallback to returning `null`.
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case final Subclass value:
///     return ...;
///   case _:
///     return null;
/// }
/// ```

@optionalTypeArgs TResult? mapOrNull<TResult extends Object?>(TResult? Function( _PlaceSearchResponse value)?  $default,){
final _that = this;
switch (_that) {
case _PlaceSearchResponse() when $default != null:
return $default(_that);case _:
  return null;

}
}
/// A variant of `when` that fallback to an `orElse` callback.
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case Subclass(:final field):
///     return ...;
///   case _:
///     return orElse();
/// }
/// ```

@optionalTypeArgs TResult maybeWhen<TResult extends Object?>(TResult Function(@JsonKey(name: 'documents')  List<LocationModel> places)?  $default,{required TResult orElse(),}) {final _that = this;
switch (_that) {
case _PlaceSearchResponse() when $default != null:
return $default(_that.places);case _:
  return orElse();

}
}
/// A `switch`-like method, using callbacks.
///
/// As opposed to `map`, this offers destructuring.
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case Subclass(:final field):
///     return ...;
///   case Subclass2(:final field2):
///     return ...;
/// }
/// ```

@optionalTypeArgs TResult when<TResult extends Object?>(TResult Function(@JsonKey(name: 'documents')  List<LocationModel> places)  $default,) {final _that = this;
switch (_that) {
case _PlaceSearchResponse():
return $default(_that.places);case _:
  throw StateError('Unexpected subclass');

}
}
/// A variant of `when` that fallback to returning `null`
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case Subclass(:final field):
///     return ...;
///   case _:
///     return null;
/// }
/// ```

@optionalTypeArgs TResult? whenOrNull<TResult extends Object?>(TResult? Function(@JsonKey(name: 'documents')  List<LocationModel> places)?  $default,) {final _that = this;
switch (_that) {
case _PlaceSearchResponse() when $default != null:
return $default(_that.places);case _:
  return null;

}
}

}

/// @nodoc
@JsonSerializable()

class _PlaceSearchResponse implements PlaceSearchModel {
  const _PlaceSearchResponse({@JsonKey(name: 'documents') required final  List<LocationModel> places}): _places = places;
  factory _PlaceSearchResponse.fromJson(Map<String, dynamic> json) => _$PlaceSearchResponseFromJson(json);

 final  List<LocationModel> _places;
@override@JsonKey(name: 'documents') List<LocationModel> get places {
  if (_places is EqualUnmodifiableListView) return _places;
  // ignore: implicit_dynamic_type
  return EqualUnmodifiableListView(_places);
}


/// Create a copy of PlaceSearchModel
/// with the given fields replaced by the non-null parameter values.
@override @JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
_$PlaceSearchResponseCopyWith<_PlaceSearchResponse> get copyWith => __$PlaceSearchResponseCopyWithImpl<_PlaceSearchResponse>(this, _$identity);

@override
Map<String, dynamic> toJson() {
  return _$PlaceSearchResponseToJson(this, );
}

@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _PlaceSearchResponse&&const DeepCollectionEquality().equals(other._places, _places));
}

@JsonKey(includeFromJson: false, includeToJson: false)
@override
int get hashCode => Object.hash(runtimeType,const DeepCollectionEquality().hash(_places));

@override
String toString() {
  return 'PlaceSearchModel(places: $places)';
}


}

/// @nodoc
abstract mixin class _$PlaceSearchResponseCopyWith<$Res> implements $PlaceSearchModelCopyWith<$Res> {
  factory _$PlaceSearchResponseCopyWith(_PlaceSearchResponse value, $Res Function(_PlaceSearchResponse) _then) = __$PlaceSearchResponseCopyWithImpl;
@override @useResult
$Res call({
@JsonKey(name: 'documents') List<LocationModel> places
});




}
/// @nodoc
class __$PlaceSearchResponseCopyWithImpl<$Res>
    implements _$PlaceSearchResponseCopyWith<$Res> {
  __$PlaceSearchResponseCopyWithImpl(this._self, this._then);

  final _PlaceSearchResponse _self;
  final $Res Function(_PlaceSearchResponse) _then;

/// Create a copy of PlaceSearchModel
/// with the given fields replaced by the non-null parameter values.
@override @pragma('vm:prefer-inline') $Res call({Object? places = null,}) {
  return _then(_PlaceSearchResponse(
places: null == places ? _self._places : places // ignore: cast_nullable_to_non_nullable
as List<LocationModel>,
  ));
}


}

// dart format on
